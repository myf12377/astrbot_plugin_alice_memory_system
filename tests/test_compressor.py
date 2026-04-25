"""
压缩器模块测试。
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory.compressor.compressor import DialogueCompressor
from memory.identity.identity import IdentityModule
from memory.settings import MemorySettings
from memory.storage.storage import MemoryStorage


class TestDialogueCompressor:
    """DialogueCompressor类的测试。"""

    @pytest.fixture
    def temp_dir(self) -> Iterator[Path]:
        """创建测试用临时目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def settings(self) -> MemorySettings:
        """创建测试配置。"""
        return MemorySettings(
            compress_model="test-model",
            compress_prompt="请将以下对话内容精简为一段摘要：\n\n{content}",
            dialogue_end_timeout=300,
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
    def identity_module(self, temp_dir: Path) -> IdentityModule:
        """创建身份模块。"""
        return IdentityModule(temp_dir)

    @pytest.fixture
    def storage(self, temp_dir: Path, settings: MemorySettings) -> MemoryStorage:
        """创建存储实例。"""
        return MemoryStorage(temp_dir, settings)

    @pytest.fixture
    def compressor(
        self,
        mock_context: MagicMock,
        identity_module: IdentityModule,
        storage: MemoryStorage,
        settings: MemorySettings,
    ) -> DialogueCompressor:
        """创建DialogueCompressor实例。"""
        return DialogueCompressor(
            mock_context,
            identity_module,
            storage,
            settings,
        )

    def test_format_dialogues(self, compressor: DialogueCompressor) -> None:
        """测试对话格式化。"""
        dialogues = ["user: Hello", "assistant: Hi there"]
        formatted = compressor._format_dialogues(dialogues)
        assert "user: Hello" in formatted
        assert "assistant: Hi there" in formatted

    def test_parse_score_valid(self, compressor: DialogueCompressor) -> None:
        """测试解析有效分数。"""
        assert compressor._parse_score("8") == 8
        assert compressor._parse_score("  5  ") == 5
        assert compressor._parse_score("The score is 9.") == 9

    def test_parse_score_boundary(self, compressor: DialogueCompressor) -> None:
        """测试解析边界分数。"""
        assert compressor._parse_score("0") == 0
        assert compressor._parse_score("10") == 10
        assert compressor._parse_score("15") == 10
        assert compressor._parse_score("-3") == 0

    def test_parse_score_invalid(self, compressor: DialogueCompressor) -> None:
        """测试解析无效响应默认为5。"""
        assert compressor._parse_score("no number here") == 5
        assert compressor._parse_score("") == 5
        assert compressor._parse_score("three") == 5

    def test_get_dialogues_empty(
        self,
        compressor: DialogueCompressor,
        storage: MemoryStorage,
    ) -> None:
        """测试获取无对话用户返回空列表。"""
        dialogues = compressor._get_dialogues("nonexistent", "2024-04-20")
        assert dialogues == []

    def test_get_dialogues_filters_by_date(
        self,
        compressor: DialogueCompressor,
        storage: MemoryStorage,
    ) -> None:
        """测试对话按日期过滤。"""
        item = storage.append_dialogue("user123", "user", "Message on target day")
        date_obj = datetime.strptime("2024-04-20", "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        date_ts = date_obj.timestamp()
        storage.update_l1_dialogue_timestamp("user123", item.message_id, date_ts)

        dialogues = compressor._get_dialogues("user123", "2024-04-20")
        assert len(dialogues) >= 1

    @pytest.mark.asyncio
    async def test_compress_day_no_dialogues(
        self,
        compressor: DialogueCompressor,
    ) -> None:
        """测试无对话时压缩返回None。"""
        result = await compressor.compress_day("nonexistent", "2024-04-20")
        assert result is None

    @pytest.mark.asyncio
    async def test_compress_day_success(
        self,
        compressor: DialogueCompressor,
        mock_context: MagicMock,
        storage: MemoryStorage,
    ) -> None:
        """测试成功压缩一天对话。"""
        item1 = storage.append_dialogue("user123", "user", "Hello")
        item2 = storage.append_dialogue("user123", "assistant", "Hi")

        date_obj = datetime.strptime("2024-04-20", "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        date_ts = date_obj.timestamp()
        storage.update_l1_dialogue_timestamp("user123", item1.message_id, date_ts)
        storage.update_l1_dialogue_timestamp("user123", item2.message_id, date_ts)

        mock_context.llm_generate.return_value = "Summary of conversation"

        result = await compressor.compress_day("user123", "2024-04-20")
        assert result == "Summary of conversation"
        assert mock_context.llm_generate.call_count == 2

        summaries = storage.get_l2_summaries()
        assert len(summaries) == 1
        assert summaries[0].summary == "Summary of conversation"

    @pytest.mark.asyncio
    async def test_compress_day_with_custom_model(
        self,
        mock_context: MagicMock,
        identity_module: IdentityModule,
        storage: MemoryStorage,
        settings: MemorySettings,
    ) -> None:
        """测试使用自定义模型压缩。"""
        settings.compress_model = "custom-compress-model"
        compressor = DialogueCompressor(
            mock_context,
            identity_module,
            storage,
            settings,
        )
        item = storage.append_dialogue("user123", "user", "Test")

        date_obj = datetime.strptime("2024-04-20", "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        date_ts = date_obj.timestamp()
        storage.update_l1_dialogue_timestamp("user123", item.message_id, date_ts)

        mock_context.llm_generate.return_value = "Summary"

        await compressor.compress_day("user123", "2024-04-20")
        call_kwargs = mock_context.llm_generate.call_args
        assert call_kwargs[1]["generate_config"]["model"] == "custom-compress-model"
