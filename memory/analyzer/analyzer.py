"""
分析器模块 - 基于LLM的重要性分析。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory.settings import MemorySettings


class ImportanceAnalyzer:
    """基于LLM分析内容重要性。

    属性:
        settings: 记忆配置。
    """

    def __init__(
        self,
        context: Any,
        settings: MemorySettings,
    ) -> None:
        """初始化重要性分析器。

        Args:
            context: 具有llm_generate能力的AstrBot上下文。
            settings: 记忆配置。
        """
        self._context = context
        self._settings = settings
        self._model = settings.importance_analyze_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    async def analyze(self, content: str) -> int:
        """分析内容的重要性分数。

        Args:
            content: 要分析的内容。

        Returns:
            重要性分数 (0-10)。
        """
        prompt = self._build_prompt(content)
        response = await self._call_llm(prompt)
        score = self._parse_score(response)
        return score

    def _build_prompt(self, content: str) -> str:
        """构建分析提示词。

        Args:
            content: 要分析的内容。

        Returns:
            格式化后的提示词字符串。
        """
        return (
            "请分析以下内容的重要性，评分范围为0-10分。\n"
            "0分：完全无关紧要的信息，如问候、闲聊\n"
            "5分：一般重要的信息，有一定保留价值\n"
            "10分：极其重要的信息，如个人偏好、关键决策、长期目标等\n\n"
            f"内容：{content}\n\n"
            "请只输出一个0-10的数字分数，不要有其他文字。"
        )

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM进行分析。

        Args:
            prompt: 发送给LLM的提示词。

        Returns:
            LLM响应文本。
        """
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
        return response

    def _parse_score(self, response: str) -> int:
        """从LLM响应中解析重要性分数。

        Args:
            response: LLM响应文本。

        Returns:
            解析后的分数 (0-10)，解析失败默认为0。
        """
        cleaned = response.strip()
        match = re.search(r"-?\d+", cleaned)
        if match:
            score = int(match.group())
            return max(0, min(10, score))
        return 0

    async def should_promote_to_l3(self, content: str) -> bool:
        """检查内容是否应升级到L3。

        Args:
            content: 要检查的内容。

        Returns:
            如果重要性分数>=阈值则返回True。
        """
        score = await self.analyze(content)
        return score >= self._settings.importance_threshold
