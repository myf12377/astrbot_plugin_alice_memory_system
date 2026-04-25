"""
压缩器模块 - 每日对话摘要。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory.identity.identity import IdentityModule
    from memory.settings import MemorySettings
    from memory.storage.storage import MemoryStorage


class DialogueCompressor:
    """使用LLM处理每日对话摘要。

    属性:
        identity_module: 用于用户解析的身份模块。
        storage: 记忆存储实例。
        settings: 记忆配置。
    """

    def __init__(
        self,
        context: Any,
        identity_module: IdentityModule,
        storage: MemoryStorage,
        settings: MemorySettings,
    ) -> None:
        """初始化对话压缩器。

        Args:
            context: 具有llm_generate能力的AstrBot上下文。
            identity_module: 用于用户解析的身份模块。
            storage: 记忆存储实例。
            settings: 记忆配置。
        """
        self._context = context
        self._identity_module = identity_module
        self._storage = storage
        self._settings = settings
        self._model = settings.compress_model
        self._prompt_template = settings.compress_prompt
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    async def compress_day(
        self,
        user_id: str,
        date: str,
        hidden: bool = False,
    ) -> str | None:
        """将用户某一天的对话压缩为L2摘要。

        Args:
            user_id: 用户标识符。
            date: 日期字符串 (YYYY-MM-DD)。
            hidden: 是否隐藏摘要。

        Returns:
            生成的摘要，如果没有对话则返回None。
        """
        dialogues = self._get_dialogues(user_id, date)
        if not dialogues:
            return None

        content = self._format_dialogues(dialogues)
        summary = await self._generate_summary(content)
        importance = await self._estimate_importance(summary)

        self._storage.add_summary(date, summary, importance, hidden=hidden)
        return summary

    def _get_dialogues(
        self,
        user_id: str,
        date: str,
    ) -> list[str]:
        """获取指定日期的对话。

        Args:
            user_id: 用户标识符。
            date: 日期字符串 (YYYY-MM-DD)。

        Returns:
            格式化后的对话字符串列表。
        """
        start_date = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = start_date + timedelta(days=1)
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()

        dialogues = self._storage.get_l1_dialogues(user_id)
        filtered = [d for d in dialogues if start_ts <= d.timestamp < end_ts]
        return [f"{d.role}: {d.content}" for d in filtered]

    def _format_dialogues(self, dialogues: list[str]) -> str:
        """格式化对话以供摘要。

        Args:
            dialogues: 对话字符串列表。

        Returns:
            格式化后的对话内容。
        """
        return "\n".join(dialogues)

    async def _generate_summary(self, content: str) -> str:
        """使用LLM生成摘要。

        Args:
            content: 要摘要的对话内容。

        Returns:
            生成的摘要。
        """
        prompt = self._prompt_template.format(content=content)
        model = self._model if self._model else None
        generate_config: dict[str, Any] = {
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if model:
            generate_config["model"] = model

        response: str = await self._context.llm_generate(
            prompt, generate_config=generate_config
        )
        return response.strip()

    async def _estimate_importance(self, summary: str) -> int:
        """评估摘要的重要性。

        Args:
            summary: 生成的摘要。

        Returns:
            估计的重要性分数 (0-10)。
        """
        prompt = (
            "请评估以下摘要的重要性，评分范围为0-10分。\n"
            "0分：完全无关紧要\n"
            "5分：一般重要\n"
            "10分：极其重要\n\n"
            f"摘要：{summary}\n\n"
            "请只输出一个0-10的数字分数，不要有其他文字。"
        )

        generate_config: dict[str, Any] = {
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._model:
            generate_config["model"] = self._model

        response: str = await self._context.llm_generate(
            prompt, generate_config=generate_config
        )
        return self._parse_score(response)

    def _parse_score(self, response: str) -> int:
        """从LLM响应中解析重要性分数。

        Args:
            response: LLM响应文本。

        Returns:
            解析后的分数 (0-10)，解析失败默认为5。
        """
        cleaned = response.strip()
        match = re.search(r"-?\d+", cleaned)
        if match:
            score = int(match.group())
            return max(0, min(10, score))
        return 5
