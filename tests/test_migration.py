"""
迁移模块测试。
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from memory.migration.migration import MigrationModule
from memory.settings import MemorySettings
from memory.storage.storage import MemoryStorage


class TestMigrationModule:
    """MigrationModule类的测试。"""

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
    def migration(self, temp_dir: Path, settings: MemorySettings) -> MigrationModule:
        """创建迁移模块实例。"""
        return MigrationModule(temp_dir, settings)

    @pytest.fixture
    def storage(self, temp_dir: Path, settings: MemorySettings) -> MemoryStorage:
        """创建存储实例。"""
        return MemoryStorage(temp_dir, settings)

    def test_export_astrmem(
        self, migration: MigrationModule, storage: MemoryStorage
    ) -> None:
        """测试导出 .astrmem 文件。"""
        storage.append_dialogue("user1", "user", "Hello")
        storage.append_dialogue("user1", "assistant", "Hi")
        storage.add_summary("user1", "2024-04-20", "Test summary", 5)
        storage.add_l3_memory("user1", "Important memory")

        output_path = Path(tempfile.mktemp(suffix=".astrmem"))
        result = migration.export_astrmem("user1", output_path)

        assert result["l1_count"] == 2
        assert result["l2_count"] == 1
        assert result["l3_count"] == 1
        assert result["total"] == 4
        assert output_path.exists()

    def test_import_astrmem(
        self, migration: MigrationModule, storage: MemoryStorage
    ) -> None:
        """测试导入 .astrmem 文件。"""
        storage.append_dialogue("user1", "user", "Hello")
        storage.add_summary("user1", "2024-04-20", "Test summary", 5)

        export_path = Path(tempfile.mktemp(suffix=".astrmem"))
        migration.export_astrmem("user1", export_path)

        storage.delete_l1_dialogue(
            "user1", storage.get_l1_dialogues("user1")[0].message_id
        )
        storage.delete_l2_summary(
            storage.get_l2_summaries()[0].summary_id, storage.get_l2_summaries()[0].date
        )

        result = migration.import_astrmem("user1", export_path)

        assert result["l1_count"] == 1
        assert result["l2_count"] == 1
        assert result["l3_count"] == 0

    def test_import_astrmem_user_id_mismatch(
        self, migration: MigrationModule, storage: MemoryStorage
    ) -> None:
        """测试导入时用户ID不匹配。"""
        storage.append_dialogue("user1", "user", "Hello")

        export_path = Path(tempfile.mktemp(suffix=".astrmem"))
        migration.export_astrmem("user1", export_path)

        with pytest.raises(ValueError, match="用户ID不匹配"):
            migration.import_astrmem("user2", export_path)

    def test_export_chroma(
        self, migration: MigrationModule, storage: MemoryStorage
    ) -> None:
        """测试导出 ChromaDB 格式。"""
        storage.add_l3_memory("user1", "Memory 1")
        storage.add_l3_memory("user1", "Memory 2")

        output_dir = Path(tempfile.mkdtemp())
        result = migration.export_chroma("user1", output_dir)

        assert result["l3_count"] == 2
        assert Path(result["output_path"]).exists()

    def test_import_chroma(
        self, migration: MigrationModule, storage: MemoryStorage
    ) -> None:
        """测试导入 ChromaDB 格式。"""
        storage.add_l3_memory("user1", "Memory 1")

        export_dir = Path(tempfile.mkdtemp())
        migration.export_chroma("user1", export_dir)

        export_file = export_dir / "user1_l3.json"
        migration.import_chroma("user1", export_file)

        memories = storage.get_l3_memories("user1")
        assert len(memories) == 2

    def test_backup(self, migration: MigrationModule, storage: MemoryStorage) -> None:
        """测试备份功能。"""
        storage.append_dialogue("user1", "user", "Hello")
        storage.add_summary("user1", "2024-04-20", "Summary", 5)

        backup_dir = Path(tempfile.mkdtemp())
        result = migration.backup(backup_dir)

        assert "backup_path" in result
        assert result["l1_count"] == 1
        assert result["l2_count"] == 1

    def test_restore(self, migration: MigrationModule, storage: MemoryStorage) -> None:
        """测试恢复功能。"""
        storage.append_dialogue("user1", "user", "Hello")
        storage.add_summary("user1", "2024-04-20", "Summary", 5)

        backup_dir = Path(tempfile.mkdtemp())
        result = migration.backup(backup_dir)
        backup_path = Path(result["backup_path"])

        storage.delete_l1_dialogue(
            "user1", storage.get_l1_dialogues("user1")[0].message_id
        )

        result = migration.restore(backup_path)

        assert result["l1_count"] == 1
        assert result["l2_count"] == 1
