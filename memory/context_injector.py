"""
记忆上下文注入器 — 四管线独立注入。

L1  → request.contexts（无标记，自然消失）
L2-A → extra_user_content_parts（[周摘要]，覆盖式）
L2-B → extra_user_content_parts（[L2记忆]，覆盖式）
L3  → extra_user_content_parts（[L3记忆]，覆盖式）
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

if TYPE_CHECKING:
    from memory.identity.identity import IdentityModule
    from memory.plugin_config import PluginConfig
    from memory.storage.storage import MemoryStorage
    from memory.vector_store.vector_store import VectorStore


# 上下文中的记忆标记（管线级自主覆盖）
L2_PATH_A_MARKER = "[周摘要]"
L2_PATH_B_MARKER = "[L2记忆]"
L3_MARKER = "[L3记忆]"


class ContextInjector:
    """记忆上下文注入器 — 四管线独立管理。

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
        self, user_id: str, request: ProviderRequest,
    ) -> None:
        """按 config 开关调度四条注入管线。"""
        if self._config.inject_l1:
            await self.inject_l1(user_id, request)
        if self._config.inject_l2_path_a:
            await self.inject_l2_path_a(user_id, request)
        if self._config.inject_l2_path_b:
            await self.inject_l2_path_b(user_id, request)
        if self._config.inject_l3:
            await self.inject_l3(user_id, request)

    # ==================================================================
    # L1 — 日内原始对话
    # ==================================================================

    async def inject_l1(
        self, user_id: str, request: ProviderRequest,
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
    # L2 Path A — 渐进周摘要
    # ==================================================================

    async def inject_l2_path_a(
        self, user_id: str, request: ProviderRequest,
    ) -> None:
        """注入周摘要到 extra_user_content_parts [周摘要]。

        周一跳过（Scheduler 凌晨已清空）。
        """
        if self._is_monday():
            return

        weekly = self._storage.get_weekly_summary(user_id)
        if not weekly or not weekly.get("summary"):
            return

        self._clean_marker(request, L2_PATH_A_MARKER)
        request.extra_user_content_parts.append(
            TextPart(text=(
                f"{L2_PATH_A_MARKER}\n"
                f"以下为本周期概括总结（日细节将由 [L2记忆] 提供）：\n"
                f"{weekly['summary']}"
            )),
        )

    # ==================================================================
    # L2 Path B — 每日磁盘摘要
    # ==================================================================

    async def inject_l2_path_b(
        self, user_id: str, request: ProviderRequest,
    ) -> None:
        """注入最近 N 天日摘要到 extra_user_content_parts [L2记忆]（周一不跳过）。"""
        summaries = self._storage.get_daily_summaries(
            user_id, last=self._config.l2_daily_inject_count,
        )
        if not summaries:
            return

        combined = "\n".join(
            f"[{s.date}] {s.summary}" for s in summaries if not s.hidden
        )
        if not combined:
            return

        self._clean_marker(request, L2_PATH_B_MARKER)
        request.extra_user_content_parts.append(
            TextPart(text=(
                f"{L2_PATH_B_MARKER}\n"
                f"以下为近期每日详细记录（周概括见 [周摘要]）：\n"
                f"{combined}"
            )),
        )

    # ==================================================================
    # L3 — 长期向量记忆
    # ==================================================================

    async def inject_l3(
        self, user_id: str, request: ProviderRequest,
    ) -> None:
        """语义检索 L3 记忆，注入到 extra_user_content_parts [L3记忆]。"""
        if not self._vector_store:
            return

        query = getattr(request, "prompt", "") or ""
        if not query:
            return

        # 修复：参数顺序 → search(user_id, query, top_k)
        results = await self._vector_store.search(user_id, query, top_k=3)

        self._clean_marker(request, L3_MARKER)

        threshold = self._config.l3_merge_similarity
        for r in results:
            score = r.get("distance", 0)
            # distance 越低越相似（cosine distance = 1 - similarity）
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
        return datetime.now(timezone.utc).weekday() == 0

    @staticmethod
    def _get_week_start() -> str:
        """获取本周一的日期字符串（UTC）。"""
        now = datetime.now(timezone.utc)
        monday = now - timedelta(days=now.weekday())
        return monday.strftime("%Y-%m-%d")

    @staticmethod
    def _clean_marker(request: ProviderRequest, marker: str) -> None:
        """移除 extra_user_content_parts 中以指定 marker 开头的旧内容。"""
        request.extra_user_content_parts = [
            p for p in request.extra_user_content_parts
            if not getattr(p, "text", "").startswith(marker)
        ]

    # ==================================================================
    # 纯读取方法 — 供主动层/中间层调用（不操作 req）
    # ==================================================================

    def get_l1_context(self, user_id: str, limit: int | None = None) -> str | None:
        """读取 L1 最近 N 轮对话，返回格式化文本。

        返回格式:
            [L1记忆]
            用户: ...
            助手: ...
        """
        rounds = self._storage.get_recent_rounds(user_id)
        if not rounds:
            return None

        if limit is not None:
            rounds = rounds[-limit:]

        lines = ["[L1记忆]"]
        for msg in rounds:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                continue  # 跳过日期标记
            role_label = "用户" if role == "user" else "助手"
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines)

    def get_l2_path_a_context(self, user_id: str) -> str | None:
        """读取 Path A 渐进周摘要，返回格式化文本。

        返回格式:
            [L2周摘要]
            <summary>
        """
        weekly = self._storage.get_weekly_summary(user_id)
        if not weekly or not weekly.get("summary"):
            return None

        return f"[L2周摘要]\n{weekly['summary']}"

    def get_l2_path_b_context(
        self, user_id: str, days: int | None = None
    ) -> str | None:
        """读取 Path B 近 N 天日摘要，返回格式化文本。

        返回格式:
            [L2日摘要]
            [2026-05-07] <summary>
            [2026-05-06] <summary>
        """
        limit = days if days is not None else self._config.l2_daily_inject_count
        summaries = self._storage.get_daily_summaries(user_id, last=limit)
        if not summaries:
            return None

        week_start = self._get_week_start()
        parts: list[str] = []
        for s in summaries:
            if s.hidden:
                continue
            if s.date >= week_start:
                continue  # 本周的跳过（周摘要覆盖）
            parts.append(f"[{s.date}] {s.summary}")

        if not parts:
            return None

        return "[L2日摘要]\n" + "\n".join(parts)

    async def get_l3_context(
        self, user_id: str, query: str = "", top_k: int | None = None
    ) -> str | None:
        """读取 L3 相关向量记忆，返回格式化文本。

        Args:
            query: 搜索查询。空字符串时检索最近记忆。
            top_k: 返回条数，默认使用配置值。

        返回格式:
            [L3记忆]
            1. <content>
            2. <content>
        """
        if not self._vector_store:
            return None

        k = top_k if top_k is not None else getattr(self._config, "l3_merge_similarity", 3)
        query = query.strip() if query else ""

        results = await self._vector_store.search(user_id, query, top_k=min(k, 5))
        if not results:
            return None

        threshold = self._config.l3_merge_similarity
        items: list[str] = []
        for i, r in enumerate(results, 1):
            score = r.get("distance", 0)
            similarity = 1.0 - score
            if similarity >= threshold:
                content = r.get("content", "")
                if content:
                    items.append(f"{i}. {content}")

        if not items:
            return None

        return "[L3记忆]\n" + "\n".join(items)
