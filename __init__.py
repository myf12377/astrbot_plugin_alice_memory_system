"""
astrbot_alice_memory_modul — Alice 三层记忆系统插件
提供 L1/L2/L3 三层记忆存储功能，支持 ChromaDB 向量检索与 LLM 压缩/分析。

导出内容:
    - MemorySettings: 配置模型
    - MemoryStorage: 存储模块
    - IdentityModule: 身份模块
    - VectorStore: 向量存储模块
    - ImportanceAnalyzer: 重要性分析模块
    - DialogueCompressor: 对话压缩模块
"""

from .memory import (
    DialogueCompressor,
    IdentityModule,
    ImportanceAnalyzer,
    MemorySettings,
    MemoryStorage,
    VectorStore,
)

# v1.0 旧入口已归档至 _legacy/astrbot_plugin_alice_memory_system_v1_legacy.py
# 新入口 main.py 待重构完成后创建

__all__ = [
    "DialogueCompressor",
    "IdentityModule",
    "ImportanceAnalyzer",
    "MemorySettings",
    "MemoryStorage",
    "VectorStore",
]

__version__ = "1.0.0"
