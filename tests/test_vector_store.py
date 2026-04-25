"""
向量存储模块测试。
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from memory.settings import MemorySettings
from memory.vector_store.vector_store import VectorStore


class TestVectorStore:
    """VectorStore类的测试。"""

    @pytest.fixture
    def temp_dir(self) -> Iterator[Path]:
        """创建测试用临时目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def settings(self) -> MemorySettings:
        """创建测试配置。"""
        return MemorySettings(data_dir=Path(tempfile.mkdtemp()))

    @pytest.fixture
    def vector_store(self, temp_dir: Path, settings: MemorySettings) -> VectorStore:
        """创建向量存储实例。"""
        return VectorStore(temp_dir, settings)

    def test_add_memory(self, vector_store: VectorStore) -> None:
        """测试添加记忆。"""
        memory_id = vector_store.add_memory("user1", "Test content")
        assert memory_id is not None
        assert len(memory_id) > 0

    def test_add_memory_with_metadata(
        self,
        vector_store: VectorStore,
    ) -> None:
        """测试添加带元数据的记忆。"""
        memory_id = vector_store.add_memory(
            "user1",
            "Test content",
            {"importance": 8},
        )
        assert memory_id is not None

    def test_get_user_memories(self, vector_store: VectorStore) -> None:
        """测试获取用户所有记忆。"""
        vector_store.add_memory("user1", "Memory 1")
        vector_store.add_memory("user1", "Memory 2")
        memories = vector_store.get_user_memories("user1")
        assert len(memories) == 2

    def test_get_user_memories_empty(
        self,
        vector_store: VectorStore,
    ) -> None:
        """测试获取不存在用户的记忆返回空列表。"""
        memories = vector_store.get_user_memories("nonexistent")
        assert memories == []

    def test_search(self, vector_store: VectorStore) -> None:
        """测试搜索记忆。"""
        vector_store.add_memory("user1", "Apple is a fruit")
        vector_store.add_memory("user1", "Car is a vehicle")
        results = vector_store.search("user1", "fruit", top_k=5)
        assert len(results) >= 1

    def test_search_no_results(self, vector_store: VectorStore) -> None:
        """测试搜索无结果。"""
        vector_store.add_memory("user1", "Apple is a fruit")
        results = vector_store.search("user1", "nonexistent topic", top_k=5)
        assert len(results) == 0

    def test_delete_memory(self, vector_store: VectorStore) -> None:
        """测试删除记忆。"""
        memory_id = vector_store.add_memory("user1", "To be deleted")
        result = vector_store.delete_memory(memory_id)
        assert result is True

    def test_delete_memory_not_found(
        self,
        vector_store: VectorStore,
    ) -> None:
        """测试删除不存在的记忆返回False。"""
        result = vector_store.delete_memory("nonexistent-id")
        assert result is False

    def test_delete_user_memories(self, vector_store: VectorStore) -> None:
        """测试删除用户所有记忆。"""
        vector_store.add_memory("user1", "Memory 1")
        vector_store.add_memory("user1", "Memory 2")
        count = vector_store.delete_user_memories("user1")
        assert count == 2
        memories = vector_store.get_user_memories("user1")
        assert len(memories) == 0

    def test_update_metadata(self, vector_store: VectorStore) -> None:
        """测试更新记忆元数据。"""
        memory_id = vector_store.add_memory("user1", "Test content")
        result = vector_store.update_metadata(memory_id, {"new_key": "new_value"})
        assert result is True

    def test_update_metadata_not_found(
        self,
        vector_store: VectorStore,
    ) -> None:
        """测试更新不存在记忆的元数据返回False。"""
        result = vector_store.update_metadata("nonexistent-id", {"key": "value"})
        assert result is False

    def test_close(self, vector_store: VectorStore) -> None:
        """测试关闭向量存储。"""
        vector_store.close()
        # 关闭后操作应该返回空列表
        memories = vector_store.get_user_memories("user1")
        assert memories == []
