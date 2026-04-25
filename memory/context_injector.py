"""记忆上下文注入器。"""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from astrbot.api.provider import ProviderRequest
from memory.identity.identity import IdentityModule
from memory.settings import MemorySettings
from memory.storage.storage import MemoryStorage
from memory.vector_store.vector_store import VectorStore


# 上下文中的记忆标记
L2_CONTEXT_MARKER = "[L2记忆]"
L3_CONTEXT_MARKER = "[L3记忆]"


class ContextInjector:
    """记忆上下文注入器。

    负责将 L1/L2/L3 记忆注入到 LLM 请求上下文中。
    设计原则：上下文空间最小化，只保留未压缩对话和压缩摘要。
    """

    def __init__(
        self,
        storage: MemoryStorage,
        vector_store: VectorStore | None,
        identity_module: IdentityModule,
        settings: MemorySettings,
    ) -> None:
        self._storage = storage
        self._vector_store = vector_store
        self._identity_module = identity_module
        self._settings = settings

    def _is_monday(self) -> bool:
        """检查今天是否是周一。"""
        return datetime.now(timezone.utc).weekday() == 0

    async def inject_l1(
        self, user_id: str, request: ProviderRequest
    ) -> None:
        """注入今日 L1 对话到 request.contexts。

        L1 是日内短期记忆，当日对话原始内容，不被压缩。
        每日凌晨清空。

        Args:
            user_id: 用户 ID。
            request: LLM 请求对象。
        """
        if not self._storage:
            return

        dialogues = self._storage.get_l1_dialogues(user_id)
        today = datetime.now(timezone.utc).date()

        # 注入今日所有对话（L1 不被压缩，无需过滤）
        today_dialogues = [
            d for d in dialogues
            if datetime.fromtimestamp(d.timestamp, tz=timezone.utc).date() == today
        ]

        for d in today_dialogues:
            request.contexts.append({
                "role": d.role,
                "content": d.content
            })

    async def inject_l2(
        self, user_id: str, request: ProviderRequest
    ) -> None:
        """注入 L2 摘要到上下文（覆盖式）。

        - 获取当前本周所有 L2 摘要（按日期独立储存，每个日期只有最新摘要）
        - 合并为一个整体
        - 覆盖上下文中旧的 L2 内容
        - 每周一重置上下文中的 L2

        Args:
            user_id: 用户 ID。
            request: LLM 请求对象。
        """
        if not self._storage:
            return

        # 移除旧的 L2 内容（覆盖式注入）
        request.extra_user_content_parts = [
            p for p in request.extra_user_content_parts
            if not str(getattr(p, "text", "")).startswith(L2_CONTEXT_MARKER)
        ]

        # 每周一重置：清空上下文中的 L2，等待新一周的第一次压缩
        if self._is_monday():
            return

        # 获取所有 L2 摘要（现在是按日期存储，每个日期只有一份最新摘要）
        summaries = self._storage.get_l2_summaries()
        today = datetime.now(timezone.utc).date()
        week_start = today - timedelta(days=today.weekday())

        # 获取本周所有摘要
        week_summaries = [
            s for s in summaries
            if datetime.fromisoformat(s.date).replace(tzinfo=timezone.utc) >= week_start
        ]

        if not week_summaries:
            return

        # 按日期排序并合并（L2现在按日期存储，每个日期只有最新摘要）
        sorted_dates = sorted([s.date for s in week_summaries])
        combined = "\n".join(s.summary for s in week_summaries if s.date in sorted_dates)

        # 添加新的 L2 摘要
        request.extra_user_content_parts.append({
            "type": "text",
            "text": f"{L2_CONTEXT_MARKER}\n{combined}"
        })

    async def inject_l3(
        self, user_id: str, request: ProviderRequest
    ) -> None:
        """注入与当前对话相关的 L3 记忆。

        通过向量相似度搜索，注入相关记忆。
        使用 settings 中的 l3_merge_similarity 作为相似度阈值。

        Args:
            user_id: 用户 ID。
            request: LLM 请求对象。
        """
        if not self._vector_store:
            return

        query = getattr(request, "prompt", "") or ""
        if not query:
            return

        # 使用配置中的相似度阈值
        similarity_threshold = getattr(self._settings, "l3_merge_similarity", 0.9)
        results = self._vector_store.search(query, user_id, top_k=3)

        # 移除旧的 L3 内容
        request.extra_user_content_parts = [
            p for p in request.extra_user_content_parts
            if not str(getattr(p, "text", "")).startswith(L3_CONTEXT_MARKER)
        ]

        for r in results:
            score = getattr(r, "score", 0) or 0
            if score >= similarity_threshold:
                content = getattr(r, "content", "") or ""
                if content:
                    request.extra_user_content_parts.append({
                        "type": "text",
                        "text": f"{L3_CONTEXT_MARKER}\n{content}"
                    })
