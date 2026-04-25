# LEGACY: v1.0 旧入口，已废弃。
# 该文件已从项目根目录归档至 _legacy/ 目录。
# 新入口请在重构完成后使用。
#
# 归档原因：
#   - 存储层路径改写后，记忆写入与钩子注入链路存在严重问题
#   - PluginConfig(dataclass) 与 MemorySettings(pydantic) 双配置模型冲突
#   - ContextInjector 存在致命 Bug（参数对调、getattr/dict 混用）
#   - L2 存储缺少用户隔离
#
# 归档日期：2026-04-24

"""
AstrBot Memory Plugin - Alice 三层记忆系统插件

基于 LLM 的智能记忆管理系统，支持 L1/L2/L3 三层记忆存储。

功能特性:
    - L1 原始对话: 自动存储用户与助手的对话记录
    - L2 每日摘要: 使用 LLM 将每日对话压缩为摘要
    - L3 重要记忆: 自动识别并向量化存储重要信息

作者: Alice
版本: 1.0.0
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain
from astrbot.api.provider import ProviderRequest
from astrbot.core import logger

# 导入内部模块
from . import (
    ContextInjector,
    DialogueCompressor,
    IdentityModule,
    ImportanceAnalyzer,
    MemorySettings,
    MemoryStorage,
    VectorStore,
)


# =============================================================================
# 配置项定义
# =============================================================================


@dataclass
class PluginConfig:
    """插件配置项。

    用于在 AstrBot 管理界面中配置插件参数。
    """

    # L1 记忆配置
    l1_ttl_days: int = 7  # L1 记忆保留天数
    l1_enabled: bool = True  # 是否启用 L1 记忆存储

    # L2 记忆配置
    l2_ttl_days: int = 7  # L2 记忆保留天数
    l2_enabled: bool = True  # 是否启用 L2 摘要压缩

    # L3 记忆配置
    l3_recheck_interval_days: int = 30  # L3 重评间隔天数
    l3_enabled: bool = True  # 是否启用 L3 重要记忆
    l3_embedding_provider: str = "auto"  # 向量嵌入模型: auto/chroma
    importance_threshold: int = 8  # 重要性阈值 (0-10)

    # LLM 配置
    compress_model: str = ""  # 压缩模型名称 (空则使用默认)
    importance_analyze_model: str = ""  # 重要性分析模型 (空则使用默认)
    llm_max_tokens: int = 1024  # LLM 最大 token 数
    llm_temperature: float = 0.7  # LLM 温度参数

    # 通用配置
    silent_mode: bool = False  # 静默模式 (减少日志输出)
    data_dir: str = "data/plugins/astrmemory"  # 数据存储目录 (实际路径会在插件初始化时按机器人名称进一步划分)


# =============================================================================
# 数据导出格式
# =============================================================================


@dataclass
class ExportData:
    """数据导出格式。

    用于支持两种导出格式:
    - .astrmem.json: 跨插件迁移使用
    - .json: 通用 JSON 格式备份
    """

    version: str = "1.0"  # 导出格式版本
    exported_at: str = ""  # 导出时间 (ISO 格式)
    plugin_version: str = "1.0.0"  # 插件版本
    l1_dialogues: list[dict[str, Any]] = field(default_factory=list)
    l2_summaries: list[dict[str, Any]] = field(default_factory=list)
    l3_memories: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 主插件类
# =============================================================================


class AliceMemorySystem(star.Star):
    """Alice 三层记忆系统插件主类。

    管理 L1/L2/L3 三层记忆，支持对话存储、摘要压缩、重要性分析。

    属性:
        context: AstrBot 上下文对象。
        config: 插件配置项。
        identity_module: 用户身份模块。
        storage: 记忆存储模块。
        vector_store: 向量存储模块 (ChromaDB)。
        analyzer: 重要性分析模块。
        compressor: 对话压缩模块。
    """

    def __init__(self, context: star.Context) -> None:
        """初始化插件。

        Args:
            context: AstrBot 上下文对象。
        """
        self.context = context
        self.config: PluginConfig = PluginConfig()
        self.identity_module: IdentityModule | None = None
        self.storage: MemoryStorage | None = None
        self.vector_store: VectorStore | None = None
        self.analyzer: ImportanceAnalyzer | None = None
        self.compressor: DialogueCompressor | None = None

        # 获取机器人名称，用于按机器人划分存储
        self._bot_name = self._get_bot_name()
        # 数据目录 (符合 AstrBot 官方规范: data/plugin_data/插件名/机器人名/)
        self._data_root = Path(f"data/plugin_data/alice_memory_storage/{self._bot_name}")
        self._backup_dir = self._data_root / "backups"

        # 初始化
        self._init_plugin()

    def _get_bot_name(self) -> str:
        """获取机器人名称。

        从 AstrBot 配置中读取机器人名称，用于按机器人划分存储。

        Returns:
            机器人名称，默认 "default"
        """
        try:
            # 尝试从 platform 配置中获取机器人名称
            # cmd_config.json 中 platform[0].id 即为机器人名称
            platform_config = self.context._config.get("platform")
            if platform_config and len(platform_config) > 0:
                bot_id = platform_config[0].get("id") if isinstance(platform_config[0], dict) else None
                if bot_id:
                    return bot_id
        except Exception:
            pass
        # 默认值
        return "default"

    def _init_plugin(self) -> None:
        """初始化插件模块。

        包含错误处理,确保单个模块初始化失败不会导致整个插件崩溃。
        """
        try:
            # 加载配置
            self._load_config()

            # 确保目录存在
            self._data_root.mkdir(parents=True, exist_ok=True)
            self._backup_dir.mkdir(parents=True, exist_ok=True)

            # 构建 MemorySettings
            settings = MemorySettings(
                data_dir=self._data_root,
                l1_ttl=self.config.l1_ttl_days,
                l2_ttl=self.config.l2_ttl_days,
                l3_recheck_interval=self.config.l3_recheck_interval_days,
                importance_threshold=self.config.importance_threshold,
                compress_model=self.config.compress_model,
                importance_analyze_model=self.config.importance_analyze_model,
                llm_max_tokens=self.config.llm_max_tokens,
                llm_temperature=self.config.llm_temperature,
                silent_mode=self.config.silent_mode,
            )

            # 初始化各模块
            self._init_identity_module()
            self._init_storage(settings)
            self._init_vector_store(settings)
            self._init_analyzer(settings)
            self._init_compressor(settings)
            self._init_context_injector(settings)

            logger.info("Alice Memory System 初始化成功")

        except Exception as e:
            logger.error(f"Alice Memory System 初始化失败: {e}")
            logger.debug(traceback.format_exc())

    def _init_identity_module(self) -> None:
        """初始化身份模块。"""
        try:
            self.identity_module = IdentityModule(self._data_root)
        except Exception as e:
            logger.error(f"身份模块初始化失败: {e}")

    def _init_storage(self, settings: MemorySettings) -> None:
        """初始化存储模块。

        Args:
            settings: 记忆配置对象。
        """
        try:
            self.storage = MemoryStorage(self._data_root, settings)
        except Exception as e:
            logger.error(f"存储模块初始化失败: {e}")

    def _init_vector_store(self, settings: MemorySettings) -> None:
        """初始化向量存储模块。

        Args:
            settings: 记忆配置对象。
        """
        try:
            if self.config.l3_enabled:
                embedding_func = None

                # 根据配置决定是否使用 EmbeddingProvider
                if self.config.l3_embedding_provider == "auto":
                    # 自动模式：尝试获取 AstrBot EmbeddingProvider
                    try:
                        from astrbot.core.provider.entities import ProviderType

                        provider_manager = getattr(
                            self.context, "provider_manager", None
                        )
                        if provider_manager:
                            embedding_provider = provider_manager.get_using_provider(
                                ProviderType.EMBEDDING,
                                self.context,
                            )
                            if embedding_provider:
                                # 创建异步 wrapper
                                async def get_embeddings(
                                    texts: list[str],
                                ) -> list[list[float]]:
                                    return await embedding_provider.get_embeddings(
                                        texts
                                    )

                                embedding_func = get_embeddings
                                logger.info(
                                    "已启用 AstrBot EmbeddingProvider 用于 L3 向量存储"
                                )
                    except Exception as e:
                        logger.debug(
                            f"无法获取 EmbeddingProvider，使用 ChromaDB 默认: {e}"
                        )
                elif self.config.l3_embedding_provider == "chroma":
                    # 强制使用 ChromaDB 默认 embedding
                    logger.info("L3 使用 ChromaDB 内置 embedding 模型")
                    embedding_func = None

                self.vector_store = VectorStore(
                    self._data_root,
                    settings,
                    embedding_func=embedding_func,
                )
        except Exception as e:
            logger.error(f"向量存储模块初始化失败: {e}")

    def _init_analyzer(self, settings: MemorySettings) -> None:
        """初始化重要性分析模块。

        Args:
            settings: 记忆配置对象。
        """
        try:
            self.analyzer = ImportanceAnalyzer(self.context, settings)
        except Exception as e:
            logger.error(f"重要性分析模块初始化失败: {e}")

    def _init_compressor(self, settings: MemorySettings) -> None:
        """初始化对话压缩模块。

        Args:
            settings: 记忆配置对象。
        """
        try:
            if self.identity_module and self.storage:
                self.compressor = DialogueCompressor(
                    self.context,
                    self.identity_module,
                    self.storage,
                    settings,
                )
        except Exception as e:
            logger.error(f"对话压缩模块初始化失败: {e}")

    def _init_context_injector(self, settings: MemorySettings) -> None:
        """初始化上下文注入模块。"""
        try:
            self.context_injector = ContextInjector(
                self.storage,
                self.vector_store,
                self.identity_module,
                settings,
            )
            logger.info("ContextInjector 初始化成功")
        except Exception as e:
            logger.error(f"ContextInjector 初始化失败: {e}")
            self.context_injector = None

    def _load_config(self) -> None:
        """从配置文件加载配置。"""
        config_path = self._data_root / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
            except Exception as e:
                logger.warning(f"配置文件加载失败,使用默认配置: {e}")

    def _save_config(self) -> None:
        """保存配置到文件。"""
        try:
            config_path = self._data_root / "config.json"
            data = {
                "l1_ttl_days": self.config.l1_ttl_days,
                "l2_ttl_days": self.config.l2_ttl_days,
                "l3_recheck_interval_days": self.config.l3_recheck_interval_days,
                "l3_enabled": self.config.l3_enabled,
                "l2_enabled": self.config.l2_enabled,
                "l1_enabled": self.config.l1_enabled,
                "importance_threshold": self.config.importance_threshold,
                "compress_model": self.config.compress_model,
                "importance_analyze_model": self.config.importance_analyze_model,
                "llm_max_tokens": self.config.llm_max_tokens,
                "llm_temperature": self.config.llm_temperature,
                "silent_mode": self.config.silent_mode,
                "data_dir": self.config.data_dir,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")

    def _get_user_id(self, event: AstrMessageEvent) -> str | None:
        """从事件获取用户 ID。

        Args:
            event: 消息事件对象。

        Returns:
            用户 ID 字符串,获取失败返回 None。
        """
        try:
            platform = getattr(event.message_obj, "platform", None)
            user_id = str(event.get_sender_id())
            if platform and self.identity_module:
                return self.identity_module.register_user(platform, user_id)
            return user_id
        except Exception as e:
            logger.error(f"获取用户ID失败: {e}")
            return None

    def _log_debug(self, message: str) -> None:
        """输出调试日志。

        Args:
            message: 日志消息。
        """
        if not self.config.silent_mode:
            logger.debug(f"AliceMemory: {message}")

    def _log_info(self, message: str) -> None:
        """输出信息日志。

        Args:
            message: 日志消息。
        """
        if not self.config.silent_mode:
            logger.info(f"AliceMemory: {message}")

    # =========================================================================
    # 消息处理 - L1 记忆存储
    # =========================================================================

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent) -> None:
        """消息处理 - 存储对话到 L1。

        捕获用户发送的消息并存储到 L1 记忆。

        Args:
            event: 消息事件对象。
        """
        # 检查是否启用 L1 存储
        if not self.config.l1_enabled or not self.storage:
            return

        # 获取用户 ID
        user_id = self._get_user_id(event)
        if not user_id:
            return

        try:
            # 提取消息内容
            message_parts = []
            for comp in event.get_messages():
                if isinstance(comp, Plain) and comp.text:
                    message_parts.append(comp.text)
                elif hasattr(comp, "file"):
                    message_parts.append(f"[文件: {comp.file}]")
                elif hasattr(comp, "url"):
                    message_parts.append(f"[图片: {comp.url}]")

            content = "".join(message_parts).strip()
            if not content:
                return

            # 确定消息角色
            role = "user"
            if (
                hasattr(event.message_obj, "is_from_self")
                and event.message_obj.is_from_self
            ):
                role = "assistant"

            # 存储对话
            self.storage.append_dialogue(user_id, role, content)
            self._log_debug(f"保存对话 | 用户: {user_id[:8]}... | 角色: {role}")

        except Exception as e:
            logger.error(f"保存对话失败: {e}")
            logger.debug(traceback.format_exc())

    # =========================================================================
    # LLM 响应处理 - 存储助手回复
    # =========================================================================

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp) -> None:
        """LLM 响应后 - 存储助手回复到 L1。

        Args:
            event: 消息事件对象。
            resp: LLM 响应对象。
        """
        if not self.config.l1_enabled or not self.storage:
            return

        user_id = self._get_user_id(event)
        if not user_id or not resp.completion_text:
            return

        try:
            self.storage.append_dialogue(user_id, "assistant", resp.completion_text)
            self._log_debug(f"保存助手回复 | 用户: {user_id[:8]}...")
        except Exception as e:
            logger.error(f"保存助手回复失败: {e}")

    # =========================================================================
    # LLM 请求处理 - 注入记忆上下文
    # =========================================================================

    @filter.on_llm_request()
    async def handle_memory_recall(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """[事件钩子] 在 LLM 请求前，注入记忆上下文。

        每次 LLM 请求时，根据配置注入 L1/L2/L3 记忆。
        """
        if not self.context_injector:
            return

        user_id = self._get_user_id(event)
        if not user_id:
            return

        try:
            # L1: 每次请求都注入今日对话
            if self.config.l1_enabled and self.config.inject_l1:
                await self.context_injector.inject_l1(user_id, req)

            # L2: 注入本周摘要（覆盖式，每次压缩后更新）
            if self.config.l2_enabled and self.config.inject_l2:
                await self.context_injector.inject_l2(user_id, req)

            # L3: 向量相似度检索注入
            if self.config.l3_enabled and self.config.inject_l3 and self.vector_store:
                await self.context_injector.inject_l3(user_id, req)

        except Exception as e:
            logger.error(f"注入记忆上下文失败: {e}")
            logger.debug(traceback.format_exc())

    # =========================================================================
    # 导出功能
    # =========================================================================

    async def export_data(self, user_id: str | None = None) -> dict[str, Any]:
        """导出记忆数据。

        支持两种格式:
        - astrmem: 跨插件迁移格式
        - json: 通用 JSON 格式

        Args:
            user_id: 指定用户 ID,为 None 则导出所有用户数据。

        Returns:
            导出统计信息。
        """
        try:
            export_info = ExportData()
            export_info.exported_at = datetime.now(timezone.utc).isoformat()

            # 获取用户列表
            if user_id:
                user_ids = [user_id]
            elif self.identity_module:
                user_ids = self.identity_module.get_all_users()
            else:
                user_ids = []

            # 导出各层数据
            total_l1, total_l2, total_l3 = 0, 0, 0

            for uid in user_ids:
                if self.storage:
                    # L1 对话
                    l1_dialogues = self.storage.get_l1_dialogues(uid)
                    for d in l1_dialogues:
                        export_info.l1_dialogues.append(d.to_dict())
                        total_l1 += 1

                    # L2 摘要
                    l2_summaries = self.storage.get_l2_summaries()
                    for s in l2_summaries:
                        export_info.l2_summaries.append(s.to_dict())
                        total_l2 += 1

                    # L3 记忆
                    if self.vector_store and self.config.l3_enabled:
                        l3_memories = self.vector_store.get_user_memories(uid)
                        for m in l3_memories:
                            export_info.l3_memories.append(m)
                            total_l3 += 1

            # 保存配置
            export_info.config = {
                "l1_ttl_days": self.config.l1_ttl_days,
                "l2_ttl_days": self.config.l2_ttl_days,
                "l3_recheck_interval_days": self.config.l3_recheck_interval_days,
                "importance_threshold": self.config.importance_threshold,
            }

            self._log_info(
                f"导出完成 | L1: {total_l1} | L2: {total_l2} | L3: {total_l3}"
            )

            return {
                "success": True,
                "version": export_info.version,
                "exported_at": export_info.exported_at,
                "l1_count": total_l1,
                "l2_count": total_l2,
                "l3_count": total_l3,
                "data": export_info,
            }

        except Exception as e:
            logger.error(f"导出数据失败: {e}")
            return {"success": False, "error": str(e)}

    async def import_data(
        self,
        data: dict[str, Any],
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """导入记忆数据。

        Args:
            data: 导入数据字典。
            user_id: 指定用户 ID,为 None 则使用数据中的用户 ID。

        Returns:
            导入统计信息。
        """
        try:
            version = data.get("version", "1.0")
            if version != "1.0":
                return {"success": False, "error": f"不支持的版本: {version}"}

            imported_l1, imported_l2, imported_l3 = 0, 0, 0

            if self.storage:
                # 导入 L1 对话
                for d in data.get("l1_dialogues", []):
                    self.storage.append_dialogue(
                        d.get("user_id", user_id or ""),
                        d.get("role", "user"),
                        d.get("content", ""),
                    )
                    imported_l1 += 1

                # 导入 L2 摘要
                for s in data.get("l2_summaries", []):
                    self.storage.add_summary(
                        s.get("user_id", user_id or ""),
                        s.get("date", ""),
                        s.get("summary", ""),
                        s.get("importance", 5),
                    )
                    imported_l2 += 1

                # 导入 L3 记忆
                if self.vector_store and self.config.l3_enabled:
                    for m in data.get("l3_memories", []):
                        await self.vector_store.add_memory(
                            m.get("user_id", user_id or ""),
                            m.get("content", ""),
                            m.get("metadata", {}),
                        )
                        imported_l3 += 1

            self._log_info(
                f"导入完成 | L1: {imported_l1} | L2: {imported_l2} | L3: {imported_l3}"
            )

            return {
                "success": True,
                "l1_count": imported_l1,
                "l2_count": imported_l2,
                "l3_count": imported_l3,
            }

        except Exception as e:
            logger.error(f"导入数据失败: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # 数据备份
    # =========================================================================

    async def backup(self) -> dict[str, Any]:
        """创建数据备份。

        备份存储在 data/plugins/astrmemory/{bot_name}/backups/ 目录下。

        Returns:
            备份统计信息。
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = self._backup_dir / f"backup_{timestamp}.json"

            # 导出所有数据
            export_result = await self.export_data()
            if not export_result.get("success"):
                return export_result

            # 保存备份文件
            backup_data = export_result.get("data")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)

            # 清理旧备份 (保留最近 10 个)
            self._cleanup_old_backups()

            self._log_info(f"备份完成: {backup_path}")

            return {
                "success": True,
                "backup_path": str(backup_path),
                "l1_count": export_result.get("l1_count", 0),
                "l2_count": export_result.get("l2_count", 0),
                "l3_count": export_result.get("l3_count", 0),
            }

        except Exception as e:
            logger.error(f"备份失败: {e}")
            return {"success": False, "error": str(e)}

    def _cleanup_old_backups(self, keep_count: int = 10) -> None:
        """清理旧备份文件。

        Args:
            keep_count: 保留的备份文件数量。
        """
        try:
            backups = sorted(
                self._backup_dir.glob("backup_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_backup in backups[keep_count:]:
                old_backup.unlink()
                self._log_debug(f"删除旧备份: {old_backup.name}")
        except Exception as e:
            logger.warning(f"清理旧备份失败: {e}")

    # =========================================================================
    # 统计信息
    # =========================================================================

    async def get_memory_stats(self, user_id: str) -> dict[str, Any]:
        """获取用户记忆统计。

        Args:
            user_id: 用户 ID。

        Returns:
            记忆统计信息。
        """
        try:
            if not self.storage:
                return {"success": False, "error": "存储模块未初始化"}

            l1_count = len(self.storage.get_l1_dialogues(user_id))
            l2_count = len(self.storage.get_l2_summaries())
            l3_count = len(self.storage.get_l3_memories(user_id))

            return {
                "success": True,
                "user_id": user_id,
                "l1_dialogues": l1_count,
                "l2_summaries": l2_count,
                "l3_memories": l3_count,
            }

        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # 搜索功能
    # =========================================================================

    async def search_memory(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """搜索用户记忆。

        Args:
            user_id: 用户 ID。
            query: 搜索查询。
            top_k: 返回结果数量。

        Returns:
            匹配的记忆列表。
        """
        if not self.vector_store or not self.config.l3_enabled:
            return []

        try:
            return await self.vector_store.search(user_id, query, top_k)
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return []

    # =========================================================================
    # 压缩功能
    # =========================================================================

    async def compress_day(self, user_id: str, date: str) -> str | None:
        """压缩指定日期的对话为摘要。

        Args:
            user_id: 用户 ID。
            date: 日期字符串 (YYYY-MM-DD)。

        Returns:
            生成的摘要,失败返回 None。
        """
        if not self.compressor or not self.config.l2_enabled:
            return None

        try:
            summary = await self.compressor.compress_day(user_id, date)
            if summary:
                self._log_info(f"压缩完成 | 用户: {user_id[:8]}... | 日期: {date}")
            return summary
        except Exception as e:
            logger.error(f"压缩对话失败: {e}")
            return None

    # =========================================================================
    # 命令接口
    # =========================================================================

    @filter.command("记忆压缩")
    async def cmd_compress(
        self,
        event: AstrMessageEvent,
        date: str = "",
    ) -> None:
        """手动压缩今日对话为摘要。

        Args:
            event: 消息事件对象。
            date: 可选日期字符串 (YYYY-MM-DD)，默认为今日。
        """
        user_id = self._get_user_id(event)
        if not user_id:
            await event.reply([Plain("无法获取用户信息")])
            return

        # 解析日期，默认为今日
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # 验证日期格式
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            await event.reply(
                [Plain(f"日期格式错误，请使用 YYYY-MM-DD 格式，输入: {date}")]
            )
            return

        await event.reply([Plain(f"正在压缩 {date} 的对话，请稍候...")])

        summary = await self.compress_day(user_id, date)
        if summary:
            preview = summary[:200] + ("..." if len(summary) > 200 else "")
            await event.reply([Plain(f"压缩完成！\n\n摘要预览：\n{preview}")])
        else:
            await event.reply([Plain(f"压缩失败，可能没有 {date} 的对话记录")])
