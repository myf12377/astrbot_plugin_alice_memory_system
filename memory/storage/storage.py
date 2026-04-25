"""
存储模块 - L1/L2/L3记忆存储。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from memory.settings import MemorySettings


@dataclass
class L1MemoryItem:
    """L1记忆项 - 原始对话。

    属性:
        message_id: 消息唯一标识符。
        user_id: 用户标识符。
        role: 角色（user/assistant）。
        content: 对话内容。
        timestamp: 时间戳。
        compressed: 是否已被压缩为L2摘要。
        content_type: 内容类型（text/image/audio）。
        media_url: 原始媒体URL。
    """

    message_id: str
    user_id: str
    role: str
    content: str
    timestamp: float
    compressed: bool = False
    content_type: str = "text"
    media_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "compressed": self.compressed,
            "content_type": self.content_type,
            "media_url": self.media_url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> L1MemoryItem:
        """从字典创建。"""
        return cls(
            message_id=data["message_id"],
            user_id=data["user_id"],
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"],
            compressed=data.get("compressed", False),
            content_type=data.get("content_type", "text"),
            media_url=data.get("media_url"),
        )


@dataclass
class L2SummaryItem:
    """L2记忆项 - 每日摘要。

    属性:
        summary_id: 摘要唯一标识符。
        user_id: 用户标识符。
        date: 日期字符串 (YYYY-MM-DD)。
        summary: 摘要内容。
        importance: 重要性分数 (0-10)。
        timestamp: 时间戳。
        hidden: 是否隐藏（不注入前端对话）。
    """

    summary_id: str
    user_id: str
    date: str
    summary: str
    importance: int
    timestamp: float
    hidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "summary_id": self.summary_id,
            "user_id": self.user_id,
            "date": self.date,
            "summary": self.summary,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "hidden": self.hidden,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> L2SummaryItem:
        """从字典创建。"""
        return cls(
            summary_id=data["summary_id"],
            user_id=data["user_id"],
            date=data["date"],
            summary=data["summary"],
            importance=data["importance"],
            timestamp=data["timestamp"],
            hidden=data.get("hidden", False),
        )


@dataclass
class L3MemoryItem:
    """L3记忆项 - 重要记忆向量。

    属性:
        memory_id: 记忆唯一标识符。
        user_id: 用户标识符。
        content: 记忆内容。
        metadata: 元数据。
        timestamp: 时间戳。
    """

    memory_id: str
    user_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> L3MemoryItem:
        """从字典创建。"""
        return cls(
            memory_id=data["memory_id"],
            user_id=data["user_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", 0.0),
        )


class MemoryStorage:
    """三层记忆存储。

    使用JSON文件持久化存储L1/L2/L3记忆。

    属性:
        data_dir: 数据存储根目录。
        settings: 记忆配置。
    """

    def __init__(
        self,
        data_dir: Path,
        settings: MemorySettings,
    ) -> None:
        """初始化存储。

        Args:
            data_dir: 数据存储根目录。
            settings: 记忆配置。
        """
        self._data_dir = data_dir
        self._settings = settings
        self._l1_dir = data_dir / "l1"
        self._l2_dir = data_dir / "l2"
        self._l3_dir = data_dir / "l3"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """确保目录存在。"""
        self._l1_dir.mkdir(parents=True, exist_ok=True)
        self._l2_dir.mkdir(parents=True, exist_ok=True)
        self._l3_dir.mkdir(parents=True, exist_ok=True)

    def _get_l1_path(self, user_id: str) -> Path:
        """获取用户L1存储路径。"""
        return self._l1_dir / f"{user_id}.json"

    def _get_l2_path(self, date: str) -> Path:
        """获取指定日期的L2存储路径。"""
        return self._l2_dir / f"{date}.json"

    def _get_l3_path(self, user_id: str) -> Path:
        """获取用户L3存储路径。"""
        return self._l3_dir / f"{user_id}.json"

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        """加载JSON文件。"""
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cast(list[dict[str, Any]], json.load(f))
        except (json.JSONDecodeError, IOError):
            return []

    def _save_json(self, path: Path, data: list[dict[str, Any]]) -> None:
        """保存JSON文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # L1对话操作

    def append_dialogue(
        self,
        user_id: str,
        role: str,
        content: str,
    ) -> L1MemoryItem:
        """添加对话到L1存储。

        Args:
            user_id: 用户标识符。
            role: 角色（user/assistant）。
            content: 对话内容。

        Returns:
            创建的L1记忆项。
        """
        item = L1MemoryItem(
            message_id=str(uuid.uuid4()),
            user_id=user_id,
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).timestamp(),
        )
        path = self._get_l1_path(user_id)
        data = self._load_json(path)
        data.append(item.to_dict())
        self._save_json(path, data)
        return item

    def get_l1_dialogues(self, user_id: str) -> list[L1MemoryItem]:
        """获取用户所有L1对话。

        Args:
            user_id: 用户标识符。

        Returns:
            L1记忆项列表。
        """
        path = self._get_l1_path(user_id)
        data = self._load_json(path)
        return [L1MemoryItem.from_dict(d) for d in data]

    def delete_l1_dialogue(self, user_id: str, message_id: str) -> bool:
        """删除L1对话。

        Args:
            user_id: 用户标识符。
            message_id: 消息ID。

        Returns:
            是否删除成功。
        """
        path = self._get_l1_path(user_id)
        data = self._load_json(path)
        original_len = len(data)
        data = [d for d in data if d.get("message_id") != message_id]
        if len(data) < original_len:
            self._save_json(path, data)
            return True
        return False

    def update_l1_dialogue_timestamp(
        self,
        user_id: str,
        message_id: str,
        timestamp: float,
    ) -> bool:
        """更新L1对话时间戳。

        Args:
            user_id: 用户标识符。
            message_id: 消息ID。
            timestamp: 新时间戳。

        Returns:
            是否更新成功。
        """
        path = self._get_l1_path(user_id)
        data = self._load_json(path)
        for item in data:
            if item.get("message_id") == message_id:
                item["timestamp"] = timestamp
                self._save_json(path, data)
                return True
        return False

    # L2摘要操作

    def add_summary(
        self,
        date: str,
        summary: str,
        importance: int,
        hidden: bool = False,
    ) -> L2SummaryItem:
        """添加L2摘要（覆盖式写入）。

        每次压缩生成当日最新摘要，覆盖写入同一日期文件。

        Args:
            date: 日期字符串 (YYYY-MM-DD)。
            summary: 摘要内容。
            importance: 重要性分数。
            hidden: 是否隐藏。

        Returns:
            创建的L2记忆项。
        """
        item = L2SummaryItem(
            summary_id=str(uuid.uuid4()),
            user_id="",  # L2按日期存储，不再需要user_id
            date=date,
            summary=summary,
            importance=importance,
            timestamp=datetime.now(timezone.utc).timestamp(),
            hidden=hidden,
        )
        # 覆盖写入：每次压缩生成当日最新摘要
        path = self._get_l2_path(date)
        self._save_json(path, [item.to_dict()])
        return item

    def get_l2_summaries(self, date: str | None = None) -> list[L2SummaryItem]:
        """获取L2摘要。

        Args:
            date: 可选日期字符串 (YYYY-MM-DD)。如果为None，返回所有摘要。

        Returns:
            L2记忆项列表。
        """
        if date:
            # 返回指定日期的摘要
            path = self._get_l2_path(date)
            data = self._load_json(path)
            return [L2SummaryItem.from_dict(d) for d in data]
        else:
            # 返回所有日期的摘要（遍历l2目录）
            results = []
            for f in self._l2_dir.glob("*.json"):
                data = self._load_json(f)
                results.extend([L2SummaryItem.from_dict(d) for d in data])
            return results

    def delete_l2_summary(self, summary_id: str, date: str) -> bool:
        """删除L2摘要。

        Args:
            summary_id: 摘要ID。
            date: 日期字符串 (YYYY-MM-DD)。

        Returns:
            是否删除成功。
        """
        path = self._get_l2_path(date)
        data = self._load_json(path)
        original_len = len(data)
        data = [d for d in data if d.get("summary_id") != summary_id]
        if len(data) < original_len:
            self._save_json(path, data)
            return True
        return False

    # L3重要记忆操作

    def add_l3_memory(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """添加L3重要记忆。

        Args:
            user_id: 用户标识符。
            content: 记忆内容。
            metadata: 元数据。

        Returns:
            创建的记忆ID。
        """
        memory_id = str(uuid.uuid4())
        item = L3MemoryItem(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc).timestamp(),
        )
        path = self._get_l3_path(user_id)
        data = self._load_json(path)
        data.append(item.to_dict())
        self._save_json(path, data)
        return memory_id

    def get_l3_memories(self, user_id: str) -> list[L3MemoryItem]:
        """获取用户所有L3记忆。

        Args:
            user_id: 用户标识符。

        Returns:
            L3记忆项列表。
        """
        path = self._get_l3_path(user_id)
        data = self._load_json(path)
        return [L3MemoryItem.from_dict(d) for d in data]

    def delete_l3_memory(self, user_id: str, memory_id: str) -> bool:
        """删除L3记忆。

        Args:
            user_id: 用户标识符。
            memory_id: 记忆ID。

        Returns:
            是否删除成功。
        """
        path = self._get_l3_path(user_id)
        data = self._load_json(path)
        original_len = len(data)
        data = [d for d in data if d.get("memory_id") != memory_id]
        if len(data) < original_len:
            self._save_json(path, data)
            return True
        return False

    # 对话会话管理

    def get_active_dialogue(self, user_id: str) -> list[L1MemoryItem]:
        """获取用户当前活跃对话。

        Args:
            user_id: 用户标识符。

        Returns:
            L1记忆项列表。
        """
        return self.get_l1_dialogues(user_id)

    def end_dialogue(self, user_id: str) -> None:
        """结束用户当前对话（暂无特殊处理）。

        Args:
            user_id: 用户标识符。
        """
        pass

    def save(self) -> None:
        """保存所有数据（暂无特殊处理）。"""
        pass
