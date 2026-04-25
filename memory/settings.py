"""
设置模块 - 记忆插件配置。

本模块定义了 Alice 三层记忆系统的所有配置项，包括：
- L1 原始对话：存储、检索、清理配置
- L2 每日摘要：压缩间隔、保留天数配置
- L3 重要记忆：向量存储、合并、删除阈值配置
- LLM 相关：模型选择、Token限制、温度参数配置
- 压缩反馈：手动压缩完成后的反馈模式配置

配置通过 _conf_schema.json 定义，前端界面会根据 schema 自动渲染。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MemorySettings(BaseModel):
    """记忆插件配置模型。

    所有配置项都有默认值，可以根据实际需求调整。
    配置值来自 AstrBot 的插件配置系统（_conf_schema.json）。

    属性:
        data_dir: 数据存储根目录，包含 l1/l2/l3 子目录。
        l1_ttl: L1 层记忆保留天数（按时间清理）。
        l2_ttl: L2 层记忆保留天数（按时间清理）。
        l3_recheck_interval: L3 层重新评估间隔天数。
        silent_mode: 静默模式，减少日志输出。
        compress_model: 压缩用的 LLM 模型名称，为空使用默认模型。
        compress_prompt: 压缩提示词模板，{content} 占位符会被对话内容替换。
        importance_analyze_model: 重要性分析用的 LLM 模型名称。
        importance_threshold: 重要性阈值 (0-10)，达到此分数存入 L3。
        llm_max_tokens: LLM 请求的最大 Token 数量。
        llm_temperature: LLM 温度参数，控制输出随机性 (0.0-2.0)。
        l1_retention_hours: L1 对话保留小时数，0 则使用每日清空模式。
        l1_cleanup_hour: L1 每日清空时间（小时），支持小数如 2.5 = 02:30。
        l2_compress_interval_hours: L2 压缩间隔（小时），支持小数如 8.5 = 8小时30分钟。
        l3_merge_similarity: L3 向量相似度合并阈值 (0-1)，达到此相似度则合并。
        l3_delete_threshold: L3 删除权重阈值 (0-10)，低于此值且相似则删除。
        l3_merge_interval_days: L3 合并周期（天），控制合并执行的频率。
    """

    # ==========================================================================
    # 基础配置
    # ==========================================================================

    #: 数据存储根目录，包含 l1/l2/l3 三个子目录
    data_dir: Path = Field(default=Path("data/plugins/astrmemory"))

    #: L1 记忆保留天数，超过此天数的 L1 对话会被定时清理删除
    l1_ttl: int = Field(default=7)

    #: L2 摘要保留天数，超过此天数的 L2 摘要会被定时清理删除
    l2_ttl: int = Field(default=7)

    #: L3 重要记忆重新评估的时间间隔（天）
    l3_recheck_interval: int = Field(default=30)

    #: 静默模式，开启后减少日志输出，适合生产环境
    silent_mode: bool = Field(default=False)

    # ==========================================================================
    # LLM 模型配置
    # ==========================================================================

    #: 压缩用的 LLM 模型名称，为空则使用 AstrBot 默认模型
    #: 示例: "gpt-4o-mini", "claude-3-haiku"
    compress_model: str = Field(default="")

    #: 压缩提示词模板，{content} 会被待压缩的对话内容替换
    compress_prompt: str = Field(
        default="请将以下对话内容精简为一段摘要：\n\n{content}"
    )

    #: 重要性分析用的 LLM 模型名称，为空则使用 AstrBot 默认模型
    importance_analyze_model: str = Field(default="")

    #: 重要性分数阈值 (0-10)，LLM 判定的重要性达到此分数才会存入 L3
    #: 0分 = 完全无关紧要，5分 = 一般重要，10分 = 极其重要
    importance_threshold: int = Field(default=8)

    #: LLM 请求的最大 Token 数量，控制生成内容的长度上限
    llm_max_tokens: int = Field(default=1024)

    #: LLM 温度参数，控制输出的随机性和创造性
    #: 0.0 = 确定性输出，2.0 = 高度随机输出
    #: 摘要任务建议使用 0.5-0.7，既能保持一致性又有适当变化
    llm_temperature: float = Field(default=0.7)

    # ==========================================================================
    # L1 记忆配置 - 原始对话存储
    # ==========================================================================

    #: 是否启用 L1 记忆存储
    l1_enabled: bool = Field(default=True)

    #: L1 对话保留小时数，0 表示不使用小时模式（改用每日凌晨清空）
    #: 非0值时，L1 对话会按小时数自动清理（如 24 = 24小时后删除）
    #: 注意：与 l1_cleanup_hour 冲突，0 使用时间点清空，非0 使用时间间隔清空
    l1_retention_hours: float = Field(default=0)

    #: L1 每日清空时间（小时），仅在 l1_retention_hours=0 时生效
    #: 支持小数实现精确时间：2.5 = 02:30，2.75 = 02:45
    #: 建议设置在凌晨3-5点，此时服务器负载较低
    l1_cleanup_hour: float = Field(default=2.0)

    #: 注入上下文的最大对话条数
    l1_search_limit: int = Field(default=10)

    #: 存储媒体内容描述
    store_media_content: bool = Field(default=True)

    #: 图片描述使用的模型
    media_to_text_model: str = Field(default="")

    #: 语音识别(STT)模型
    stt_model: str = Field(default="")

    # ==========================================================================
    # L2 记忆配置 - 每日摘要
    # ==========================================================================

    #: 是否启用 L2 每日摘要压缩
    l2_enabled: bool = Field(default=True)

    #: L2 压缩任务执行间隔（小时），控制多久压缩一次 L1 对话为 L2 摘要
    #: 支持小数实现精确间隔：8.5 = 8小时30分钟，0.5 = 30分钟
    #: 建议值：4-12 小时，太频繁会增加 API 消耗，太久会丢失细节
    l2_compress_interval_hours: float = Field(default=8.0)

    #: L2 摘要默认隐藏（不注入前端对话）
    #: 设置为 True 时，所有压缩产生的摘要都不会注入到前端对话中
    #: 包括定时压缩和手动压缩 /compact 命令
    l2_summary_hidden: bool = Field(default=False)

    #: 压缩时显示进度提示
    #: 设置为 False 时，"正在压缩..." 这条消息不会发送
    compact_progress_feedback: bool = Field(default=True)

    #: ==========================================================================
    # 压缩反馈配置 - 手动压缩
    #: ==========================================================================

    #: 手动压缩完成后给用户的反馈模式
    #: - silent: 彻底静默，不发送任何消息
    #: - fixed: 发送固定文本
    #: - llm: 大模型动态生成符合上下文的反馈
    #: - visible: 显示约200字符的摘要预览给用户
    manual_compress_feedback_mode: str = Field(default="llm")

    #: 手动压缩使用固定文本模式时的反馈内容
    manual_compress_feedback_text: str = Field(default="今日对话已存档 ✨")

    #: 手动压缩使用大模型动态反馈时的提示词
    #: {context_summary} 占位符会被今日对话的情绪/内容摘要替换
    manual_compress_llm_prompt: str = Field(
        default="今日对话已完成存档。请根据今日对话的情绪氛围，以自然、人类化的方式告知用户。"
        "不要透露任何具体内容，如同朋友间简单告知'今天的对话已经记下了'。"
        "语气要贴合今日对话的整体氛围。"
    )

    # ==========================================================================
    # L3 记忆配置 - 重要记忆向量
    # ==========================================================================

    #: 是否启用 L3 重要记忆向量存储
    l3_enabled: bool = Field(default=True)

    #: L3 向量嵌入模型
    #: - auto: 使用 AstrBot EmbeddingProvider（如已配置）
    #: - chroma: 使用 ChromaDB 内置模型
    l3_embedding_provider: str = Field(default="auto")

    #: L3 向量相似度合并阈值 (0-1)，当两条记忆的相似度 >= 此值时合并
    #: 合并策略：保留较早的记忆，将较新的标记为已合并
    #: 建议值：0.85-0.95，太高会降低合并效果，太低会错误合并不同记忆
    l3_merge_similarity: float = Field(default=0.9)

    #: L3 删除权重阈值 (0-10)，重要性低于此值且与其他记忆相似时会删除
    #: 删除条件：importance < threshold AND similarity >= 0.85
    #: 建议值：3-5，太高可能丢失重要记忆，太低会保留太多无关记忆
    l3_delete_threshold: float = Field(default=3.0)

    #: L3 合并任务执行周期（天），控制多久执行一次合并和删除
    #: 设为 30 表示每月执行一次 L3 记忆的合并与清理
    #: 注意：实际执行还受 last_merge 元数据控制
    l3_merge_interval_days: int = Field(default=30)

    # ==========================================================================
    # 上下文注入配置
    # ==========================================================================

    #: 是否注入 L1 到上下文
    inject_l1: bool = Field(default=True)

    #: 是否注入 L2 到上下文
    inject_l2: bool = Field(default=True)

    #: 是否注入 L3 到上下文
    inject_l3: bool = Field(default=True)

    def model_post_init(self, __dict__: dict[str, Any]) -> None:
        """初始化后处理，确保数据目录存在。

        在模型初始化完成后自动创建必要的目录结构。
        这确保了即使目录不存在，存储模块也能正常工作。

        Args:
            __dict__: Pydantic 传递给模型的参数字典。
        """
        # 创建数据根目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
