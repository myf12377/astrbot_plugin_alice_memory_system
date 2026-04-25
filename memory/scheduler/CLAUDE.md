# Scheduler Module

## 角色

5 段定时任务编排，负责记忆的日常维护。

## 状态：重构中

## 构造

```python
def __init__(
    self, context: Any, storage: MemoryStorage, identity_module: IdentityModule,
    vector_store: VectorStore | None, config: PluginConfig,
    compressor: DialogueCompressor | None = None, analyzer: ImportanceAnalyzer | None = None,
) -> None
```

## 公开 API

```python
def register_tasks(self) -> None: ...
"""向 AstrBot CronJobManager 注册 5 个定时任务。无 cron_manager 时静默返回。"""
```

5 个任务入口（均为 async，遍历全部用户执行）：

| 时间 | 方法 | 操作 |
|------|------|------|
| 01:00 | `_compress_daily()` | Storage(L1)→Compressor→Storage(L2) |
| 02:00 | `_l1_cleanup()` | Storage.delete_old_l1_dialogues |
| 03:00 | `_l3_maintenance()` | VectorStore.apply_decay→get_gray→Analyzer.batch_recheck |
| 04:00 | `_compress_context()` | Storage(L1+L2+周)→Compressor→Storage(覆写周) |
| 周一05:00 | `_reset_weekly()` | Storage.clear_weekly_summary |

## 边界

不负责：压缩/衰减/清理算法（调各模块），钩子注册（Main 的职责），命令处理。
依赖方：Main.__init__()（唯一调用方）。
