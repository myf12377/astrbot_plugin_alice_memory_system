"""
存储模块测试。
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from memory.settings import MemorySettings
from memory.storage.storage import MemoryStorage


class TestMemoryStorage:
    """MemoryStorage类的测试。"""

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
    def storage(self, temp_dir: Path, settings: MemorySettings) -> MemoryStorage:
        """创建存储实例。"""
        return MemoryStorage(temp_dir, settings)

    def test_append_dialogue(self, storage: MemoryStorage) -> None:
        """测试添加对话。"""
        item = storage.append_dialogue("user1", "user", "Hello")
        assert item.user_id == "user1"
        assert item.role == "user"
        assert item.content == "Hello"
        assert item.message_id is not None

    def test_get_l1_dialogues(self, storage: MemoryStorage) -> None:
        """测试获取L1对话。"""
        storage.append_dialogue("user1", "user", "Hello")
        storage.append_dialogue("user1", "assistant", "Hi")
        dialogues = storage.get_l1_dialogues("user1")
        assert len(dialogues) == 2

    def test_delete_l1_dialogue(self, storage: MemoryStorage) -> None:
        """测试删除L1对话。"""
        item = storage.append_dialogue("user1", "user", "Hello")
        result = storage.delete_l1_dialogue("user1", item.message_id)
        assert result is True
        dialogues = storage.get_l1_dialogues("user1")
        assert len(dialogues) == 0

    def test_delete_l1_dialogue_not_found(self, storage: MemoryStorage) -> None:
        """测试删除不存在的对话返回False。"""
        result = storage.delete_l1_dialogue("user1", "nonexistent")
        assert result is False

    def test_add_summary(self, storage: MemoryStorage) -> None:
        """测试添加L2摘要。"""
        item = storage.add_summary("user1", "2024-04-20", "Test summary", 5)
        assert item.user_id == "user1"
        assert item.date == "2024-04-20"
        assert item.summary == "Test summary"
        assert item.importance == 5

    def test_get_l2_summaries(self, storage: MemoryStorage) -> None:
        """测试获取L2摘要。"""
        storage.add_summary("2024-04-20", "Summary 1", 5)
        storage.add_summary("2024-04-21", "Summary 2", 7)
        summaries = storage.get_l2_summaries()
        assert len(summaries) == 2

    def test_delete_l2_summary(self, storage: MemoryStorage) -> None:
        """测试删除L2摘要。"""
        item = storage.add_summary("2024-04-20", "Test", 5)
        result = storage.delete_l2_summary(item.summary_id, "2024-04-20")
        assert result is True

    def test_add_l3_memory(self, storage: MemoryStorage) -> None:
        """测试添加L3记忆。"""
        memory_id = storage.add_l3_memory("user1", "Important info", {"key": "value"})
        assert memory_id is not None
        assert len(memory_id) > 0

    def test_get_l3_memories(self, storage: MemoryStorage) -> None:
        """测试获取L3记忆。"""
        storage.add_l3_memory("user1", "Memory 1")
        storage.add_l3_memory("user1", "Memory 2")
        memories = storage.get_l3_memories("user1")
        assert len(memories) == 2

    def test_delete_l3_memory(self, storage: MemoryStorage) -> None:
        """测试删除L3记忆。"""
        memory_id = storage.add_l3_memory("user1", "Important")
        result = storage.delete_l3_memory("user1", memory_id)
        assert result is True
        memories = storage.get_l3_memories("user1")
        assert len(memories) == 0

    def test_get_active_dialogue(self, storage: MemoryStorage) -> None:
        """测试获取活跃对话。"""
        storage.append_dialogue("user1", "user", "Hello")
        dialogues = storage.get_active_dialogue("user1")
        assert len(dialogues) == 1
