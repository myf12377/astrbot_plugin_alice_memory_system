"""
调度器模块测试。
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.identity.identity import IdentityModule
from memory.scheduler.scheduler import Scheduler
from memory.settings import MemorySettings
from memory.storage.storage import MemoryStorage


class TestScheduler:
    """Scheduler类的测试。"""

    @pytest.fixture
    def temp_dir(self) -> Iterator[Path]:
        """创建测试用临时目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def settings(self) -> MemorySettings:
        """创建测试配置。"""
        return MemorySettings(
            l1_ttl=7,
            l2_ttl=7,
            l3_recheck_interval=30,
            llm_max_tokens=1024,
            llm_temperature=0.7,
        )

    @pytest.fixture
    def mock_context(self) -> MagicMock:
        """创建模拟AstrBot上下文。"""
        context = MagicMock()
        context.cron_manager = MagicMock()
        context.llm_generate = AsyncMock()
        return context

    @pytest.fixture
    def identity_module(self, temp_dir: Path) -> IdentityModule:
        """创建身份模块。"""
        return IdentityModule(temp_dir)

    @pytest.fixture
    def storage(self, temp_dir: Path, settings: MemorySettings) -> MemoryStorage:
        """创建存储实例。"""
        return MemoryStorage(temp_dir, settings)

    @pytest.fixture
    def scheduler(
        self,
        mock_context: MagicMock,
        identity_module: IdentityModule,
        storage: MemoryStorage,
        settings: MemorySettings,
    ) -> Scheduler:
        """创建Scheduler实例。"""
        return Scheduler(
            mock_context,
            storage,
            identity_module,
            None,  # vector_store
            settings,
        )

    def test_register_tasks(
        self,
        scheduler: Scheduler,
        mock_context: MagicMock,
    ) -> None:
        """测试任务注册。"""
        scheduler.register_tasks()
        assert mock_context.cron_manager.add_basic_job.call_count == 3

    def test_register_tasks_no_cron_manager(
        self,
        identity_module: IdentityModule,
        storage: MemoryStorage,
        settings: MemorySettings,
    ) -> None:
        """测试无cron_manager时不注册任务（静默返回，不抛异常）。"""
        mock_context = MagicMock()
        mock_context.cron_manager = None
        scheduler = Scheduler(
            mock_context,
            storage,
            identity_module,
            None,
            settings,
        )
        # 应该静默返回，不抛异常
        scheduler.register_tasks()

    @pytest.mark.asyncio
    async def test_cleanup_l1(
        self,
        scheduler: Scheduler,
    ) -> None:
        """测试L1清理功能。"""
        result = await scheduler.cleanup_l1()
        assert result == 0  # 没有数据，返回0

    @pytest.mark.asyncio
    async def test_cleanup_l2(
        self,
        scheduler: Scheduler,
    ) -> None:
        """测试L2清理功能。"""
        result = await scheduler.cleanup_l2()
        assert result == 0  # 没有数据，返回0

    @pytest.mark.asyncio
    async def test_recheck_l3_no_vector_store(
        self,
        scheduler: Scheduler,
    ) -> None:
        """测试无向量存储时L3重评返回0。"""
        result = await scheduler.recheck_l3()
        assert result == 0
