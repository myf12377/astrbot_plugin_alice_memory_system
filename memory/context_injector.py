"""
记忆上下文注入器 — 三管线独立注入（v2.2.0）。

L1  → request.contexts（按日期分组，system 标记日期边界）
L2  → extra_user_content_parts（[L2记忆]，周摘要 + 非本周日摘要，去重合并）
L3  → extra_user_content_parts（[L3记忆]，按需语义检索）
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

if TYPE_CHECKING:
    from memory.identity.identity import IdentityModule
    from memory.plugin_config import PluginConfig
    from memory.storage.storage import MemoryStorage
    from memory.vector_store.vector_store import VectorStore


# 上下文中的记忆标记（管线级自主覆盖）
L2_MARKER = "[L2记忆]"
L3_MARKER = "[L3记忆]"


class ContextInjector:
    """记忆上下文注入器 — 三管线独立管理。

    设计原则：每条管线持有独立标记，只管理自己的内容，
    互不污染。注入只读，不写。
    """

    def __init__(
        self,
        storage: MemoryStorage,
        vector_store: VectorStore | None,
        identity_module: IdentityModule,
        config: PluginConfig,
    ) -> None:
        self._storage = storage
        self._vector_store = vector_store
        self._identity_module = identity_module
        self._config = config

    # ==================================================================
    # 统一入口
    # ==================================================================

    async def inject_all(
        self,
        user_id: str,
        request: ProviderRequest,
    ) -> None:
        """按 config 开关调度三条注入管线。

        注入顺序：L2（中期）→ L3（长期）→ L1（短期），
        短期记忆最靠近当前对话，权重最高。
        """
        if self._config.inject_l2_path_a or self._config.inject_l2_path_b:
            await self.inject_l2_merged(user_id, request)
        if self._config.inject_l3:
            await self.inject_l3(user_id, request)
        if self._config.inject_l1:
            await self.inject_l1(user_id, request)

    # ==================================================================
    # L1 — 日内原始对话（全量分组注入）
    # ==================================================================

    async def inject_l1(
        self,
        user_id: str,
        request: ProviderRequest,
    ) -> None:
        """注入最近 N 轮 L1 对话到 request.contexts。

        按日期分组，每天插入 system 日期标记。
        l1_inject_rounds=0 时跳过。
        """
        rounds = self._storage.get_recent_rounds(user_id)
        if not rounds:
            return

        for msg in rounds:
            request.contexts.append(msg)

    # ==================================================================
    # L2 — 中期记忆（周摘要 + 非本周日摘要，去重合并）
    # ==================================================================

    async def inject_l2_merged(
        self,
        user_id: str,
        request: ProviderRequest,
    ) -> None:
        """注入合并的 L2 记忆到 extra_user_content_parts [L2记忆]。

        合并逻辑：
        - 周摘要（非周一才注入，周一凌晨已清空）
        - 非本周的日摘要（避免与周摘要重复）
        """
        parts: list[str] = []

        # 周摘要（周一跳过）
        if self._config.inject_l2_path_a and not self._is_monday():
            weekly = self._storage.get_weekly_summary(user_id)
            if weekly and weekly.get("summary"):
                parts.append(f"[周摘要] {weekly['summary']}")

        # 非本周的日摘要
        if self._config.inject_l2_path_b:
            week_start = self._get_week_start()
            all_summaries = self._storage.get_daily_summaries(
                user_id,
                last=self._config.l2_daily_inject_count,
            )
            # 排除本周摘要（周摘要已包含）和隐藏项
            for s in all_summaries:
                if s.hidden:
                    continue
                if s.date >= week_start:
                    continue  # 本周的跳过（周摘要覆盖）
                parts.append(f"[{s.date}] {s.summary}")

        if not parts:
            return

        combined = "\n\n".join(parts)
        self._clean_marker(request, L2_MARKER)
        request.extra_user_content_parts.append(
            TextPart(text=f"{L2_MARKER}\n{combined}"),
        )

    # ==================================================================
    # L3 — 长期向量记忆
    # ==================================================================

    async def inject_l3(
        self,
        user_id: str,
        request: ProviderRequest,
    ) -> None:
        """语义检索 L3 记忆，注入到 extra_user_content_parts [L3记忆]。"""
        if not self._vector_store:
            return

        query = getattr(request, "prompt", "") or ""
        if not query:
            return

        results = await self._vector_store.search(user_id, query, top_k=3)

        self._clean_marker(request, L3_MARKER)

        threshold = self._config.l3_merge_similarity
        for r in results:
            score = r.get("distance", 0)
            similarity = 1.0 - score
            if similarity >= threshold:
                content = r.get("content", "")
                if content:
                    request.extra_user_content_parts.append(
                        TextPart(text=f"{L3_MARKER}\n{content}"),
                    )

    # ==================================================================
    # 工具
    # ==================================================================

    @staticmethod
    def _is_monday() -> bool:
        cst = ZoneInfo("Asia/Shanghai")
        return datetime.now(cst).weekday() == 0

    @staticmethod
    def _get_week_start() -> str:
        """获取本周一的日期字符串（CST）。"""
        cst = ZoneInfo("Asia/Shanghai")
        now = datetime.now(cst)
        monday = now - timedelta(days=now.weekday())
        return monday.strftime("%Y-%m-%d")

    @staticmethod
    def _clean_marker(request: ProviderRequest, marker: str) -> None:
        """移除 extra_user_content_parts 中以指定 marker 开头的旧内容。"""
        request.extra_user_content_parts = [
            p
            for p in request.extra_user_content_parts
            if not getattr(p, "text", "").startswith(marker)
        ]
