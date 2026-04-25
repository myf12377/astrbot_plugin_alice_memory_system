"""
分析器模块测试。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.analyzer.analyzer import ImportanceAnalyzer
from memory.settings import MemorySettings


class TestImportanceAnalyzer:
    """ImportanceAnalyzer类的测试。"""

    @pytest.fixture
    def settings(self) -> MemorySettings:
        """创建测试配置。"""
        return MemorySettings(
            importance_threshold=8,
            importance_analyze_model="test-model",
            llm_max_tokens=1024,
            llm_temperature=0.7,
        )

    @pytest.fixture
    def mock_context(self) -> MagicMock:
        """创建模拟AstrBot上下文。"""
        context = MagicMock()
        context.llm_generate = AsyncMock()
        return context

    @pytest.fixture
    def analyzer(
        self,
        mock_context: MagicMock,
        settings: MemorySettings,
    ) -> ImportanceAnalyzer:
        """创建ImportanceAnalyzer实例。"""
        return ImportanceAnalyzer(mock_context, settings)

    def test_build_prompt(self, analyzer: ImportanceAnalyzer) -> None:
        """测试提示词构建。"""
        content = "Test content"
        prompt = analyzer._build_prompt(content)
        assert "Test content" in prompt
        assert "0-10" in prompt

    def test_parse_score_valid(self, analyzer: ImportanceAnalyzer) -> None:
        """测试解析有效分数。"""
        assert analyzer._parse_score("8") == 8
        assert analyzer._parse_score("  5  ") == 5
        assert analyzer._parse_score("The score is 9.") == 9

    def test_parse_score_boundary(self, analyzer: ImportanceAnalyzer) -> None:
        """测试解析边界分数。"""
        assert analyzer._parse_score("0") == 0
        assert analyzer._parse_score("10") == 10
        assert analyzer._parse_score("15") == 10
        assert analyzer._parse_score("-3") == 0

    def test_parse_score_invalid(self, analyzer: ImportanceAnalyzer) -> None:
        """测试解析无效响应。"""
        assert analyzer._parse_score("no number here") == 0
        assert analyzer._parse_score("") == 0
        assert analyzer._parse_score("three") == 0

    @pytest.mark.asyncio
    async def test_analyze_success(
        self,
        analyzer: ImportanceAnalyzer,
        mock_context: MagicMock,
    ) -> None:
        """测试成功分析。"""
        mock_context.llm_generate.return_value = "8"
        score = await analyzer.analyze("Important personal preference")
        assert score == 8
        mock_context.llm_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_with_custom_model(
        self,
        mock_context: MagicMock,
        settings: MemorySettings,
    ) -> None:
        """测试使用自定义模型设置进行分析。"""
        settings.importance_analyze_model = "custom-model"
        analyzer = ImportanceAnalyzer(mock_context, settings)
        mock_context.llm_generate.return_value = "7"
        await analyzer.analyze("Test content")
        call_kwargs = mock_context.llm_generate.call_args
        assert call_kwargs[1]["generate_config"]["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_analyze_empty_model_uses_default(
        self,
        mock_context: MagicMock,
        settings: MemorySettings,
    ) -> None:
        """测试空模型时使用默认配置。"""
        settings.importance_analyze_model = ""
        analyzer = ImportanceAnalyzer(mock_context, settings)
        mock_context.llm_generate.return_value = "5"
        await analyzer.analyze("Test content")
        call_kwargs = mock_context.llm_generate.call_args
        assert "model" not in call_kwargs[1]["generate_config"]

    @pytest.mark.asyncio
    async def test_should_promote_to_l3(self, analyzer: ImportanceAnalyzer) -> None:
        """测试L3升级检查。"""
        mock_context = analyzer._context
        mock_context.llm_generate.return_value = "9"
        result = await analyzer.should_promote_to_l3("Any content")
        assert result is True
