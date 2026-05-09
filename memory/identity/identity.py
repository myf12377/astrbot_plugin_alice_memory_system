"""
身份模块 - 跨平台用户身份映射。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, cast


class IdentityModule:
    """跨平台用户身份映射。

    将不同平台的用户ID映射到统一的内部用户ID。

    属性:
        data_dir: 数据存储目录。
    """

    def __init__(self, data_dir: Path) -> None:
        """初始化身份模块。

        Args:
            data_dir: 数据存储目录。
        """
        self._data_dir = data_dir
        self._mapping_file = data_dir / "identity_mapping.json"
        self._links_file = data_dir / "user_links.json"
        self._ensure_files()

    def _ensure_files(self) -> None:
        """确保数据文件存在。"""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        if not self._mapping_file.exists():
            self._save_mapping({})
        if not self._links_file.exists():
            self._save_links({})

    def _load_mapping(self) -> dict[str, str]:
        """加载身份映射。"""
        with open(self._mapping_file, "r", encoding="utf-8") as f:
            return cast(dict[str, str], json.load(f))

    def _save_mapping(self, mapping: dict[str, str]) -> None:
        """保存身份映射。"""
        with open(self._mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    def _load_links(self) -> dict[str, list[str]]:
        """加载用户链接。"""
        with open(self._links_file, "r", encoding="utf-8") as f:
            return cast(dict[str, list[str]], json.load(f))

    def _save_links(self, links: dict[str, list[str]]) -> None:
        """保存用户链接。"""
        with open(self._links_file, "w", encoding="utf-8") as f:
            json.dump(links, f, ensure_ascii=False, indent=2)

    def register_user(
        self,
        platform: str,
        platform_user_id: str,
    ) -> str:
        """注册新用户并返回内部用户ID。

        Args:
            platform: 平台名称（如 qqofficial, aiocqhttp）。
            platform_user_id: 平台用户ID。

        Returns:
            内部用户ID。
        """
        mapping = self._load_mapping()
        key = f"{platform}:{platform_user_id}"

        if key in mapping:
            return mapping[key]

        user_id = str(uuid.uuid4())
        mapping[key] = user_id
        self._save_mapping(mapping)
        return user_id

    def get_user_id(
        self,
        platform: str,
        platform_user_id: str,
    ) -> str | None:
        """获取内部用户ID。

        Args:
            platform: 平台名称。
            platform_user_id: 平台用户ID。

        Returns:
            内部用户ID，如果不存在则返回None。
        """
        mapping = self._load_mapping()
        key = f"{platform}:{platform_user_id}"
        return mapping.get(key)

    def get_all_users(self) -> list[str]:
        """获取所有已注册的用户ID。

        Returns:
            用户ID列表。
        """
        mapping = self._load_mapping()
        return list(set(mapping.values()))

    def link_users(
        self,
        user_id: str,
        linked_user_id: str,
    ) -> bool:
        """链接两个用户账户。

        Args:
            user_id: 用户ID。
            linked_user_id: 要链接的用户ID。

        Returns:
            是否链接成功。
        """
        if user_id == linked_user_id:
            return False

        links = self._load_links()

        if user_id not in links:
            links[user_id] = []
        if linked_user_id not in links[user_id]:
            links[user_id].append(linked_user_id)

        if linked_user_id not in links:
            links[linked_user_id] = []
        if user_id not in links[linked_user_id]:
            links[linked_user_id].append(user_id)

        self._save_links(links)
        return True

    def get_linked_users(self, user_id: str) -> list[str]:
        """获取用户链接的所有账户。

        Args:
            user_id: 用户ID。

        Returns:
            链接的用户ID列表。
        """
        links = self._load_links()
        return links.get(user_id, [])

    def parse_platform_from_context(self, context: Any) -> tuple[str, str] | None:
        """从上下文解析平台信息。

        Args:
            context: AstrBot消息上下文。

        Returns:
            (平台, 用户ID) 元组，解析失败返回None。
        """
        try:
            session = context.get("session")
            if session is None:
                return None

            platform = getattr(session, "platform", None)
            user_id = getattr(session, "user_id", None)

            if platform and user_id:
                return (platform, user_id)
            return None
        except (AttributeError, TypeError):
            return None
