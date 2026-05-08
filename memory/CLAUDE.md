# memory 子包

完整架构见 [根 CLAUDE.md](../CLAUDE.md) 和 [plan 文件](.claude/plans/playful-dancing-dongarra.md)。

| 目录 | 模块 | 层 | 状态 |
|------|------|----|------|
| [identity/](identity/) | IdentityModule | 1 | 稳定 |
| [storage/](storage/) | MemoryStorage | 1 | ✅ 完成 |
| [vector_store/](vector_store/) | VectorStore | 1 | ✅ 完成 |
| [analyzer/](analyzer/) | ImportanceAnalyzer | 1 | ✅ 完成 |
| [compressor/](compressor/) | DialogueCompressor | 2 | ✅ 完成 |
| context_injector.py | ContextInjector | 3 | ✅ v2.3.2 三管线注入 + 纯读取方法（get_l1/l2/l3_context） |
| [scheduler/](scheduler/) | Scheduler | 4 | ✅ 完成 |
| plugin_config.py | PluginConfig | 0 | ✅ 完成（39字段） |
| _conf_schema.json | 框架配置 schema | — | ✅ 完成 |
| main.py | AliceMemoryPlugin | 5 | ✅ v2.3.2 完成（4命令+4种反馈模式+6公开property） |

模块边界见各自 CLAUDE.md。
