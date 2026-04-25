"""
调度器模块 - 定时任务管理。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

    from memory.identity.identity import IdentityModule
    from memory.settings import MemorySettings
    from memory.storage.storage import MemoryStorage
    from memory.vector_store.vector_store import VectorStore


class Scheduler:
    """记忆定时任务调度器。

    管理L1/L2清理和L3重新评估任务。

    属性:
        storage: 记忆存储实例。
        identity_module: 身份模块。
        vector_store: 向量存储实例（可选）。
        settings: 记忆配置。
    """

    def __init__(
        self,
        context: Any,
        storage: MemoryStorage,
        identity_module: IdentityModule,
        vector_store: VectorStore | None,
        settings: MemorySettings,
    ) -> None:
        """初始化调度器。

        Args:
            context: AstrBot上下文，包含cron_manager。
            storage: 记忆存储实例。
            identity_module: 身份模块。
            vector_store: 向量存储实例（可选）。
            settings: 记忆配置。
        """
        self._context = context
        self._storage = storage
        self._identity_module = identity_module
        self._vector_store = vector_store
        self._settings = settings
        self._l1_ttl_days = settings.l1_ttl
        self._l2_ttl_days = settings.l2_ttl
        self._l3_recheck_days = settings.l3_recheck_interval

    def register_tasks(self) -> None:
        """注册所有定时任务到AstrBot的CronJobManager。"""
        cron_manager = self._context.cron_manager
        if cron_manager is None:
            return

        # L1清理任务 - 每天凌晨2点
        cron_manager.add_basic_job(
            name="记忆_L1清理",
            cron_expression="0 2 * * *",
            handler=self._cleanup_l1_wrapper,
            description="清理超过7天的L1原始对话",
            timezone="Asia/Shanghai",
            persistent=True,
        )

        # L2清理任务 - 每天凌晨3点
        cron_manager.add_basic_job(
            name="记忆_L2清理",
            cron_expression="0 3 * * *",
            handler=self._cleanup_l2_wrapper,
            description="清理超过7天的L2每日摘要",
            timezone="Asia/Shanghai",
            persistent=True,
        )

        # L3重评任务 - 每30天
        cron_manager.add_basic_job(
            name="记忆_L3重评",
            cron_expression="0 4 * * *",
            handler=self._recheck_l3_wrapper,
            description="重新评估L3重要记忆的重要性",
            timezone="Asia/Shanghai",
            persistent=True,
        )

    def _cleanup_l1_wrapper(self) -> None:
        """L1清理任务的异步包装器。"""
        import asyncio

        asyncio.create_task(self.cleanup_l1())

    def _cleanup_l2_wrapper(self) -> None:
        """L2清理任务的异步包装器。"""
        import asyncio

        asyncio.create_task(self.cleanup_l2())

    def _recheck_l3_wrapper(self) -> None:
        """L3重评任务的异步包装器。"""
        import asyncio

        asyncio.create_task(self.recheck_l3())

    async def cleanup_l1(self) -> int:
        """清理过期L1记忆。

        Returns:
            删除的记忆条数。
        """
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._l1_ttl_days)
        cutoff_ts = cutoff.timestamp()

        deleted_count = 0
        all_users = self._identity_module.get_all_users()

        for user_id in all_users:
            dialogues = self._storage.get_l1_dialogues(user_id)
            for dialogue in dialogues:
                if dialogue.timestamp < cutoff_ts:
                    self._storage.delete_l1_dialogue(user_id, dialogue.message_id)
                    deleted_count += 1

        return deleted_count

    async def cleanup_l2(self) -> int:
        """清理过期L2记忆。

        Returns:
            删除的记忆条数。
        """
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._l2_ttl_days)
        cutoff_ts = cutoff.timestamp()

        deleted_count = 0
        all_users = self._identity_module.get_all_users()

        for user_id in all_users:
            summaries = self._storage.get_l2_summaries()
            for summary in summaries:
                if summary.timestamp < cutoff_ts:
                    self._storage.delete_l2_summary(summary.summary_id, summary.date)
                    deleted_count += 1

        return deleted_count

    async def recheck_l3(self) -> int:
        """重新评估L3记忆的重要性。

        Returns:
            更新后的记忆条数。
        """
        if self._vector_store is None:
            return 0

        updated_count = 0
        all_users = self._identity_module.get_all_users()

        for user_id in all_users:
            memories = self._vector_store.get_user_memories(user_id)
            for memory in memories:
                vector_id = memory.get("id")
                if vector_id is None:
                    continue
                content = memory.get("content", "")
                if not content:
                    continue

                # 重新评估重要性
                importance = await self._analyze_importance(content)
                if memory.get("importance") != importance:
                    # 更新元数据
                    metadata = memory.get("metadata", {})
                    metadata["importance"] = importance
                    metadata["last_recheck"] = datetime.now().isoformat()
                    self._vector_store.update_metadata(vector_id, metadata)
                    updated_count += 1

        return updated_count

    async def _analyze_importance(self, content: str) -> int:
        """分析内容重要性。

        Args:
            content: 要分析的内容。

        Returns:
            重要性分数 (0-10)。
        """
        from memory.analyzer.analyzer import ImportanceAnalyzer

        analyzer = ImportanceAnalyzer(self._context, self._settings)
        return await analyzer.analyze(content)
