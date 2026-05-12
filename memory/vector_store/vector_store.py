"""
向量存储模块 — ChromaDB 向量存储。

管理 L3 重要记忆的嵌入、衰减、合并。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from astrbot.api import logger

if TYPE_CHECKING:
    from memory.plugin_config import PluginConfig


class VectorStore:
    """ChromaDB 向量存储 — L3 重要记忆。

    支持衰减模型、灰区检测、相似合并。
    """

    def __init__(
        self,
        data_dir: Path,
        config: PluginConfig,
        embedding_func: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        """初始化向量存储。

        Args:
            data_dir: 数据存储根目录。
            config: 插件配置（PluginConfig）。
            embedding_func: 外部 embedding 函数。
                为 None 时无嵌入能力（测试兼容），生产环境始终传 EmbeddingResolver。
        """
        self._config = config
        self._embedding_func = embedding_func
        # 外部 provider 用独立 collection（避免维度锁定冲突）
        self._collection_name = (
            "astrmemory_l3_ext" if embedding_func is not None else "astrmemory_l3"
        )
        self._client: Any = None
        self._collection: Any = None
        # 延迟迁移状态
        self._migration_pending = False
        self._old_collection_data: dict | None = None
        # collection 重建回调（维度变化时通知外部执行 JSON 恢复）
        self._on_collection_rebuilt: Callable[[], Awaitable[None]] | None = None
        # 自校准标记（避免重复校准）
        self._calibrated = False
        self._init_client(data_dir)

    def _init_client(self, data_dir: Path) -> None:
        persist_dir = data_dir / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # 确保 collection 使用 cosine 距离（代码中 similarity = 1.0 - distance 依赖此假设）
        self._ensure_cosine_collection()
        # 不立即迁移 — 因为 __init__ 时 EmbeddingProvider 可能尚未加载
        # 改为标记迁移待处理，延迟到首次 L3 操作时执行 _ensure_migrated()
        self._check_migration()

    def _ensure_cosine_collection(self) -> None:
        """确保 collection 使用 cosine 距离度量且有 embedding_dim 标记。

        ChromaDB 默认 l2 距离，但代码中 similarity = 1.0 - distance
        仅对 cosine 距离有效。若现有 collection 非 cosine 或缺少
        embedding_dim（P14 前创建的旧集合），删掉重建。
        数据由 P11 的 _recover_l3_from_json 从 l3/{uid}.json 恢复。
        """
        desired_space = "cosine"
        try:
            existing = self._client.get_collection(name=self._collection_name)
            existing_space = existing.metadata.get("hnsw:space", "")
            has_dim = bool(existing.metadata.get("embedding_dim", ""))
            if existing_space == desired_space:
                # 外部 provider 必须有 embedding_dim 以支持维度变化检测
                # 内置 ChromaDB collection (embedding_func=None) 不需要
                if self._embedding_func is None or has_dim:
                    self._collection = existing
                    return
            # 距离度量不匹配 或 旧 collection 缺少 embedding_dim → 删除重建
            if not has_dim:
                logger.info(
                    "[AliceMemory] 旧 collection 缺少 embedding_dim，删除重建以确保维度一致"
                )
            else:
                logger.info(
                    "[AliceMemory] 距离度量不匹配 | current=%s → desired=%s | 删除旧 collection 重建",
                    existing_space or "l2(default)", desired_space,
                )
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass  # collection 不存在，正常创建

        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={
                "description": "AstrBot L3 memory storage",
                "hnsw:space": desired_space,
                "embedding_dim": "",  # 首次嵌入调用时填入实际维度
            },
        )

    def set_rebuild_callback(
        self, cb: Callable[[], Awaitable[None]],
    ) -> None:
        """注册 collection 重建后的恢复回调（P16：维度变化时触发 JSON 恢复）。"""
        self._on_collection_rebuilt = cb

    # ------------------------------------------------------------------
    # 延迟迁移 — 旧 ChromaDB 内置数据 → 外部 provider
    # ------------------------------------------------------------------

    def _check_migration(self) -> None:
        """检测旧 ChromaDB 内置 collection 是否有待迁移数据。

        找到数据后标记 _migration_pending=True，不立即执行迁移
        （因为 __init__ 时 EmbeddingProvider 可能尚未加载）。
        迁移完成后删除旧 collection，后续重启不再重复迁移。
        """
        if self._embedding_func is None:
            return  # 无外部 provider，无法迁移
        try:
            old = self._client.get_collection(name="astrmemory_l3")
            old_data = old.get(include=["documents", "metadatas"])
            if old_data["ids"]:
                self._migration_pending = True
                self._old_collection_data = old_data
                logger.info(
                    "[AliceMemory] 检测到旧 ChromaDB L3 数据 (%d 条)，将在首次 L3 操作时自动迁移",
                    len(old_data["ids"]),
                )
        except Exception:
            pass  # 旧 collection 不存在，正常

    async def _ensure_migrated(self) -> None:
        """如有待迁移的旧 ChromaDB 数据，执行迁移。

        在 add_memory() / search() 等异步方法开头调用，
        确保迁移在 Provider 就绪后执行。
        """
        if not self._migration_pending or self._old_collection_data is None:
            return
        self._migration_pending = False
        logger.info("[AliceMemory] 开始迁移旧 ChromaDB L3 数据...")
        try:
            await self._reindex_async(self._old_collection_data)
            self._client.delete_collection("astrmemory_l3")
            self._old_collection_data = None
        except Exception as e:
            logger.error("[AliceMemory] L3 旧数据迁移失败: %s", e)
            self._migration_pending = True  # 下次重试

    # ------------------------------------------------------------------
    # embedding 工具
    # ------------------------------------------------------------------

    async def _call_embedding_func_async(
        self, texts: list[str],
    ) -> list[list[float]]:
        """调用 embedding 函数（兼容同步/异步/类实例 __call__），并检测维度变化。"""
        if self._embedding_func is None:
            return []
        import inspect
        func = self._embedding_func
        if (
            inspect.iscoroutinefunction(func)
            or inspect.iscoroutinefunction(getattr(func, "__call__", None))
        ):
            result = await func(texts)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, func, texts)
        # 检测并记录嵌入维度，维度变化时触发重建
        if result:
            await self._track_embedding_dim(len(result[0]))
        return result

    async def _track_embedding_dim(self, actual_dim: int) -> None:
        """记录嵌入维度；若与 collection 记录不符则删除重建（JSON 恢复兜底）。"""
        meta = self._collection.metadata or {}
        stored = meta.get("embedding_dim", "")
        if not stored:
            # 首次嵌入 → 记录维度
            try:
                self._collection.modify(
                    metadata={**meta, "embedding_dim": str(actual_dim)},
                )
            except Exception:
                pass
            return
        try:
            stored_dim = int(stored)
        except (ValueError, TypeError):
            return
        if stored_dim == actual_dim:
            return
        # 维度变化 → 删除旧 collection 重建（数据由 _recover_l3_from_json 恢复）
        logger.info(
            "[AliceMemory] 嵌入维度变化 | %d→%d | 删除旧 collection 重建",
            stored_dim, actual_dim,
        )
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={
                "description": "AstrBot L3 memory storage",
                "hnsw:space": "cosine",
                "embedding_dim": str(actual_dim),
            },
        )
        logger.info("[AliceMemory] Collection 已重建 | dim=%d", actual_dim)
        if self._on_collection_rebuilt:
            await self._on_collection_rebuilt()

    # ------------------------------------------------------------------
    # 自校准 — 换模型自动计算建议阈值（P17）
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    async def _auto_calibrate(self) -> None:
        """恢复完成后自动校准阈值：两两相似度中位数 → metadata。

        原理：异主题记忆间的相似度天然低于同主题记忆。
        取两两相似度中位数作为阈值，可有效区分相关/无关。
        已校准则跳过（_calibrated 标记），避免重复计算。
        """
        if self._calibrated:
            return
        self._calibrated = True
        all_data = self._collection.get(include=["embeddings"])
        emb_list = all_data.get("embeddings")
        if emb_list is None:
            return
        embeddings = list(emb_list)
        n = len(embeddings)
        if n < 2:
            logger.info("[AliceMemory] L3 自校准跳过 | 记忆数=%d（需 ≥2 条）", n)
            return
        # 两两计算余弦相似度
        similarities: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_sim(embeddings[i], embeddings[j])
                similarities.append(round(sim, 4))
        similarities.sort()
        median = similarities[len(similarities) // 2]
        # 写入 collection metadata
        try:
            self._collection.modify(metadata={
                **self._collection.metadata,
                "similarity_threshold": f"{median:.4f}",
            })
        except Exception:
            pass
        logger.info(
            "[AliceMemory] L3 自校准完成 | 中位数=%.4f | 对数=%d | 阈值已写入 collection",
            median, len(similarities),
        )

    def get_effective_threshold(self) -> float:
        """获取当前生效的检索相似度阈值（P19 拆分后仅供搜索/注入使用）。

        优先级：用户手动覆盖（config≠默认0.4）→ 自校准值 → 默认0.4。
        供 context_injector / main.py 调用。
        合并阈值使用独立字段 l3_merge_similarity，与此无关。
        """
        # 用户手动修改了 WebUI 配置 → 优先使用
        if self._config.l3_search_similarity != 0.4:
            return self._config.l3_search_similarity
        # 自校准值
        try:
            stored = self._collection.metadata.get("similarity_threshold", "")
            if stored:
                return float(stored)
        except Exception:
            pass
        return 0.4

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_collection(self) -> bool:
        if self._collection is None:
            return False
        return True

    # ==================================================================
    # 索引重建
    # ==================================================================

    async def _reindex_async(self, old_data: dict) -> None:
        """异步用新 provider 重算向量并写回（collection 已在 _init_client 中重建）。"""
        try:
            batch_size = 50
            total = len(old_data["ids"])
            for i in range(0, total, batch_size):
                batch_ids = old_data["ids"][i : i + batch_size]
                batch_docs = old_data["documents"][i : i + batch_size]
                batch_meta = old_data["metadatas"][i : i + batch_size]
                new_vecs = await self._call_embedding_func_async(batch_docs)
                self._collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=new_vecs,
                    metadatas=batch_meta,
                )
            logger.info(
                "[AliceMemory] L3 索引重建完成 | 条目=%d", total,
            )
        except Exception:
            logger.error("[AliceMemory] L3 索引重建失败", exc_info=True)

    # ==================================================================
    # CRUD
    # ==================================================================

    async def add_memory(
        self, user_id: str, content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加记忆到向量存储。

        Returns:
            vector_id (UUID 字符串)。
        """
        if not self._ensure_collection():
            raise RuntimeError("向量存储未初始化")

        await self._ensure_migrated()

        vector_id = str(uuid.uuid4())
        now = self._now_iso()

        doc_metadata: dict[str, Any] = {
            "user_id": user_id,
            "content": content,
            "created_at": now,
            "last_accessed_at": now,
            "access_count": 0,
            "importance": metadata.get("importance", 0) if metadata else 0,
        }
        if metadata:
            doc_metadata.update(metadata)

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
        self, user_id: str, query: str, top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """语义搜索用户记忆（同时更新访问计数）。

        Returns:
            [{"id", "content", "metadata", "distance"}, ...]
        """
        if not self._ensure_collection():
            return []

        await self._ensure_migrated()

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
        ids_to_update: list[str] = []
        new_metadatas: list[dict[str, Any]] = []

        if results["ids"] and results["ids"][0]:
            now = self._now_iso()
            for i, vid in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                # 更新访问计数
                acc = meta.get("access_count", 0) + 1
                meta["access_count"] = acc
                meta["last_accessed_at"] = now
                ids_to_update.append(vid)
                new_metadatas.append(meta)

                memories.append({
                    "id": vid,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": meta,
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })

            # 批量更新访问信息
            if ids_to_update:
                self._collection.update(ids=ids_to_update, metadatas=new_metadatas)

        return memories

    def delete_memory(self, vector_id: str) -> bool:
        """按 vector_id 或前缀删除记忆（兼容 /forget 传入的短 ID）。"""
        if not self._ensure_collection():
            return False
        try:
            # 先精确匹配（完整 UUID）
            existing = self._collection.get(ids=[vector_id])
            if existing["ids"]:
                self._collection.delete(ids=[vector_id])
                return True
            # 前缀匹配（/forget 传入前 8 位 UUID）
            all_data = self._collection.get(include=[])
            for full_id in all_data.get("ids", []):
                if full_id.startswith(vector_id):
                    self._collection.delete(ids=[full_id])
                    return True
            return False
        except Exception:
            return False

    def get_user_memories(self, user_id: str) -> list[dict[str, Any]]:
        """获取用户全部记忆。"""
        if not self._ensure_collection():
            return []
        results = self._collection.get(where={"user_id": user_id})
        memories: list[dict[str, Any]] = []
        if results["ids"]:
            for i, vid in enumerate(results["ids"]):
                memories.append({
                    "id": vid,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return memories

    def delete_user_memories(self, user_id: str) -> int:
        """删除用户全部记忆。返回删除数。"""
        if not self._ensure_collection():
            return 0
        memories = self.get_user_memories(user_id)
        count = len(memories)
        if count > 0:
            self._collection.delete(where={"user_id": user_id})
        return count

    def update_metadata(self, vector_id: str, metadata: dict[str, Any]) -> bool:
        """更新记忆元数据（合并到现有元数据）。"""
        if not self._ensure_collection():
            return False
        try:
            existing = self._collection.get(ids=[vector_id])
            if not existing["ids"]:
                return False
            old = existing["metadatas"][0] if existing["metadatas"] else {}
            new = {**old, **metadata}
            self._collection.update(ids=[vector_id], metadatas=[new])
            return True
        except Exception:
            return False

    # ==================================================================
    # 衰减模型
    # ==================================================================

    def apply_decay(self, user_id: str) -> tuple[int, int]:
        """对用户全部记忆执行衰减计算。

        公式:
            days = (now - created_at).days
            effective = importance × (decay_rate ^ days)
                      + min(access_count, 10) × access_bonus

        规则:
            effective < delete_threshold  → 删除
            effective ∈ [delete_threshold, gray_zone_upper] → 灰区
            effective > gray_zone_upper → 保留

        Returns:
            (deleted_count, gray_zone_count)
        """
        if not self._ensure_collection():
            return (0, 0)

        decay_rate = self._config.l3_decay_rate
        access_bonus = self._config.l3_access_bonus
        delete_threshold = self._config.l3_delete_threshold
        gray_zone_upper = self._config.l3_gray_zone_upper

        memories = self.get_user_memories(user_id)
        if not memories:
            return (0, 0)

        now = datetime.now(timezone.utc)
        to_delete: list[str] = []
        to_update: list[str] = []
        new_metadatas: list[dict[str, Any]] = []
        gray_count = 0

        for m in memories:
            meta = m["metadata"]
            importance = float(meta.get("importance", 0))
            created_str = meta.get("created_at", "")
            access_count = int(meta.get("access_count", 0))

            # 计算创建天数
            try:
                created_at = datetime.fromisoformat(created_str)
                days = max((now - created_at).days, 0)
            except (ValueError, TypeError):
                days = 0

            effective = (importance * (decay_rate ** days)
                         + min(access_count, 10) * access_bonus)

            if effective < delete_threshold:
                to_delete.append(m["id"])
            elif effective <= gray_zone_upper:
                gray_count += 1
                meta["effective_score"] = round(effective, 4)
                to_update.append(m["id"])
                new_metadatas.append(meta)
            else:
                # 保留，更新 effective_score
                meta["effective_score"] = round(effective, 4)
                to_update.append(m["id"])
                new_metadatas.append(meta)

        # 批量删除
        for vid in to_delete:
            try:
                self._collection.delete(ids=[vid])
            except Exception:
                pass

        # 批量更新 effective_score
        if to_update:
            try:
                self._collection.update(ids=to_update, metadatas=new_metadatas)
            except Exception:
                pass

        return (len(to_delete), gray_count)

    def get_gray_zone_memories(self, user_id: str) -> list[dict[str, Any]]:
        """获取灰区记忆（effective_score ∈ [delete_threshold, gray_zone_upper]）。

        必须先调用 apply_decay 以设置 effective_score。
        """
        if not self._ensure_collection():
            return []
        delete_threshold = self._config.l3_delete_threshold
        gray_zone_upper = self._config.l3_gray_zone_upper

        all_memories = self.get_user_memories(user_id)
        gray: list[dict[str, Any]] = []
        for m in all_memories:
            score = m["metadata"].get("effective_score")
            if score is not None and delete_threshold <= score <= gray_zone_upper:
                gray.append(m)
        return gray

    # ==================================================================
    # 相似度 & 合并
    # ==================================================================

    # P20：单用户合并逻辑，供 scheduler 和 /l3_merge 命令共用
    async def merge_similar_for_user(
        self, user_id: str, analyzer: Any, threshold: float,
    ) -> tuple[int, list[dict[str, Any]]]:
        """贪心法合并用户 L3 相似记忆（P20+P21）。

        Args:
            user_id: 目标用户。
            analyzer: ImportanceAnalyzer 实例（提供 merge_content LLM 调用）。
            threshold: 合并相似度阈值（l3_merge_similarity）。

        Returns:
            (合并对数, [{"old_ids": [...], "content": ..., "score": ...}, ...])
            调用方用 merge_details 同步 JSON 存储。
        """
        memories = self.get_user_memories(user_id)
        if len(memories) < 2:
            return 0, []
        memories.sort(
            key=lambda m: m["metadata"].get("importance", 0),
            reverse=True,
        )
        consumed: set[str] = set()
        merged_count = 0
        merge_details: list[dict[str, Any]] = []

        for m1 in memories:
            if m1["id"] in consumed:
                continue
            similar = await self.search(user_id, m1["content"], top_k=5)
            for s in similar:
                if s["id"] in consumed or s["id"] == m1["id"]:
                    continue
                distance = s.get("distance", 1.0)
                if 1.0 - distance < threshold:
                    continue
                merged = await analyzer.merge_content(
                    m1["content"], s["content"],
                )
                if not merged:
                    continue
                new_score = min(
                    max(
                        m1["metadata"].get("importance", 0),
                        s["metadata"].get("importance", 0),
                    ) + 0.5,
                    10.0,  # P19 上限
                )
                new_id, old_ids = await self.merge_memories(
                    m1["id"], s["id"], merged, new_score,
                )
                merge_details.append({
                    "old_ids": old_ids,
                    "content": merged,
                    "score": new_score,
                    "new_vector_id": new_id,
                })
                consumed.add(m1["id"])
                consumed.add(s["id"])
                merged_count += 1
                break

        return merged_count, merge_details

    # ==================================================================
    # 相似度 & 合并（旧方法）
    # ==================================================================

    def find_similar(
        self, user_id: str, embedding: list[float], threshold: float,
    ) -> list[dict[str, Any]]:
        """查找与给定向量相似的用户记忆。

        Args:
            user_id: 用户标识符。
            embedding: 查询向量。
            threshold: 余弦相似度阈值（如 0.9）。

        Returns:
            相似度 ≥ threshold 的记忆列表（按相似度降序）。
        """
        if not self._ensure_collection():
            return []
        if not embedding:
            return []
        total = self._collection.count()
        if total == 0:
            return []

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(20, total),
            where={"user_id": user_id},
        )

        similar: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i, vid in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                # ChromaDB cosine: distance = 1 - similarity
                similarity = 1.0 - distance
                if similarity >= threshold:
                    similar.append({
                        "id": vid,
                        "content": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "similarity": round(similarity, 4),
                    })

        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar

    async def merge_memories(
        self, vid1: str, vid2: str, merged_content: str, new_score: float,
    ) -> tuple[str, list[str]]:
        """合并两条记忆：删旧建新（P21 返回旧 ID 供 JSON 同步）。

        Args:
            vid1, vid2: 被合并的两条旧 vector_id。
            merged_content: 合并后的内容（由 Analyzer.merge_content 生成）。
            new_score: 新分数 = max(s1, s2) + 0.5。

        Returns:
            (新 vector_id, [被删除的旧 vector_id 列表])。
        """
        # 收集旧元数据
        meta1: dict[str, Any] = {}
        meta2: dict[str, Any] = {}
        existing = self._collection.get(ids=[vid1, vid2])
        if existing["metadatas"]:
            if existing["metadatas"][0]:
                meta1 = existing["metadatas"][0]
            if len(existing["metadatas"]) > 1 and existing["metadatas"][1]:
                meta2 = existing["metadatas"][1]

        # 删除旧条目
        self._collection.delete(ids=[vid1, vid2])

        # 创建新条目
        new_id = str(uuid.uuid4())
        now = self._now_iso()
        new_metadata = {
            "user_id": meta1.get("user_id", meta2.get("user_id", "")),
            "content": merged_content,
            "created_at": now,
            "last_accessed_at": now,
            "access_count": 0,
            "importance": new_score,
            "merged_from": [vid1, vid2],
        }

        vector: list[float] | None = None
        if self._embedding_func:
            vectors = await self._call_embedding_func_async([merged_content])
            vector = vectors[0] if vectors else None

        self._collection.add(
            ids=[new_id],
            documents=[merged_content],
            metadatas=[new_metadata],
            embeddings=[vector] if vector else None,
        )
        return new_id, [vid1, vid2]  # P21 返回旧 ID 供调用方同步 JSON

    # P21：写入前去重 — 供 /important 和 L3 晋升使用
    async def add_or_merge(
        self, user_id: str, content: str, score: float,
        analyzer: Any, merge_threshold: float,
    ) -> dict[str, Any]:
        """写入前去重：相似度高则合并，否则新增。

        流程：嵌入新内容 → 查找相似记忆 → 有则先暂存再合并，无则直接存。

        Returns:
            {"action": "added"|"merged", "vector_id": str,
             "merged_content": str, "new_score": float, "old_ids": [...]}
        """
        vectors = await self._call_embedding_func_async([content])
        if not vectors:
            vid = await self.add_memory(user_id, content, {"importance": score})
            return {"action": "added", "vector_id": vid,
                    "merged_content": content, "new_score": score, "old_ids": []}
        similar = self.find_similar(user_id, vectors[0], merge_threshold)
        if not similar:
            vid = await self.add_memory(user_id, content, {"importance": score})
            return {"action": "added", "vector_id": vid,
                    "merged_content": content, "new_score": score, "old_ids": []}
        # 有相似记忆 → 先暂存新内容，再与相似条目合并
        best = similar[0]
        temp_id = await self.add_memory(user_id, content, {"importance": score})
        merged = await analyzer.merge_content(best["content"], content)
        merged_content = merged if merged else content
        new_score = min(
            max(best["metadata"].get("importance", 0), score) + 0.5, 10.0,
        )
        new_id, old_ids = await self.merge_memories(
            best["id"], temp_id, merged_content, new_score,
        )
        return {"action": "merged", "vector_id": new_id,
                "merged_content": merged_content, "new_score": new_score,
                "old_ids": old_ids}

    # ==================================================================
    # 工具
    # ==================================================================

    def get_all_users(self) -> list[str]:
        """获取所有在 ChromaDB 中有数据的用户 ID（去重排序）。"""
        if not self._ensure_collection():
            return []
        results = self._collection.get()
        users: set[str] = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                uid = meta.get("user_id", "")
                if uid:
                    users.add(uid)
        return sorted(users)

    def close(self) -> None:
        """关闭向量存储连接。"""
        self._client = None
        self._collection = None
