"""
向量存储模块 - ChromaDB向量存储。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

if TYPE_CHECKING:
    from memory.settings import MemorySettings


class VectorStore:
    """ChromaDB向量存储。

    用于存储和搜索L3重要记忆。

    属性:
        data_dir: 数据存储目录。
        settings: 记忆配置。
    """

    def __init__(
        self,
        data_dir: Path,
        settings: MemorySettings,
        embedding_func: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        """初始化向量存储。

        Args:
            data_dir: 数据存储目录。
            settings: 记忆配置。
            embedding_func: 外部 embedding 函数，用于生成向量。
                接受文本列表，返回向量列表。
                如果为 None，则使用 ChromaDB 内置的 embedding 函数。
        """
        self._data_dir = data_dir
        self._settings = settings
        self._embedding_func = embedding_func
        self._collection_name = "astrmemory_l3"
        self._client: Any = None
        self._collection: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """初始化ChromaDB客户端。"""
        persist_dir = self._data_dir / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "AstrBot L3 memory storage"},
        )

    async def _call_embedding_func_async(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """调用 embedding 函数（异步版本）。

        支持同步和异步函数。

        Args:
            texts: 文本列表。

        Returns:
            向量列表。
        """
        if self._embedding_func is None:
            return []

        import inspect

        func = self._embedding_func
        if inspect.iscoroutinefunction(func):
            # 异步函数：直接 await
            return await func(texts)
        else:
            # 同步函数：在线程池中运行
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, texts)

    async def add_memory(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加记忆到向量存储。

        Args:
            user_id: 用户标识符。
            content: 记忆内容。
            metadata: 附加元数据。

        Returns:
            向量ID。
        """
        if self._collection is None:
            raise RuntimeError("向量存储未初始化")

        import uuid

        vector_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        doc_metadata: dict[str, Any] = {
            "user_id": user_id,
            "content": content,
            "created_at": now,
            "importance": metadata.get("importance", 0) if metadata else 0,
        }
        if metadata:
            doc_metadata.update(metadata)

        # 使用外部 embedding 函数生成向量（如有）
        vector: list[float] | None = None
        if self._embedding_func:
            vectors = await self._call_embedding_func_async([content])
            vector = vectors[0] if vectors else None

        self._collection.add(
            ids=[vector_id],
            documents=[content],
            metadatas=[doc_metadata],
            embeddings=[vector] if vector else None,
        )

        return vector_id

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """搜索用户相关记忆。

        Args:
            user_id: 用户标识符。
            query: 搜索查询。
            top_k: 返回结果数量。

        Returns:
            匹配的记忆列表。
        """
        if self._collection is None:
            return []

        # 使用外部 embedding 函数生成查询向量（如有）
        query_vector: list[float] | None = None
        query_texts: list[str] | None = None
        if self._embedding_func:
            query_vectors = await self._call_embedding_func_async([query])
            query_vector = query_vectors[0] if query_vectors else None
        else:
            query_texts = [query]

        results = self._collection.query(
            query_texts=query_texts,
            query_embeddings=[query_vector] if query_vector else None,
            n_results=top_k,
            where={"user_id": user_id},
        )

        memories: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i, vector_id in enumerate(results["ids"][0]):
                memory: dict[str, Any] = {
                    "id": vector_id,
                    "content": results["documents"][0][i]
                    if results["documents"]
                    else "",
                    "metadata": results["metadatas"][0][i]
                    if results["metadatas"]
                    else {},
                    "distance": results["distances"][0][i]
                    if results["distances"]
                    else 0.0,
                }
                memories.append(memory)

        return memories

    def delete_memory(self, vector_id: str) -> bool:
        """删除指定记忆。

        Args:
            vector_id: 向量ID。

        Returns:
            是否删除成功。
        """
        if self._collection is None:
            return False

        try:
            existing = self._collection.get(ids=[vector_id])
            if not existing["ids"]:
                return False
            self._collection.delete(ids=[vector_id])
            return True
        except Exception:
            return False

    def get_user_memories(self, user_id: str) -> list[dict[str, Any]]:
        """获取用户所有记忆。

        Args:
            user_id: 用户标识符。

        Returns:
            记忆列表。
        """
        if self._collection is None:
            return []

        results = self._collection.get(
            where={"user_id": user_id},
        )

        memories: list[dict[str, Any]] = []
        if results["ids"]:
            for i, vector_id in enumerate(results["ids"]):
                memory: dict[str, Any] = {
                    "id": vector_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                }
                memories.append(memory)

        return memories

    def delete_user_memories(self, user_id: str) -> int:
        """删除用户所有记忆。

        Args:
            user_id: 用户标识符。

        Returns:
            删除的记忆数量。
        """
        if self._collection is None:
            return 0

        memories = self.get_user_memories(user_id)
        count = len(memories)

        if count > 0:
            self._collection.delete(where={"user_id": user_id})

        return count

    def update_metadata(
        self,
        vector_id: str,
        metadata: dict[str, Any],
    ) -> bool:
        """更新记忆元数据。

        Args:
            vector_id: 向量ID。
            metadata: 新的元数据。

        Returns:
            是否更新成功。
        """
        if self._collection is None:
            return False

        try:
            existing = self._collection.get(ids=[vector_id])
            if not existing["ids"]:
                return False

            old_metadata = existing["metadatas"][0] if existing["metadatas"] else {}
            new_metadata = {**old_metadata, **metadata}

            self._collection.update(
                ids=[vector_id],
                metadatas=[new_metadata],
            )
            return True
        except Exception:
            return False

    def close(self) -> None:
        """关闭向量存储连接。"""
        self._client = None
        self._collection = None
