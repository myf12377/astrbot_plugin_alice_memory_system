# Migration Module

## 概述

迁移模块负责数据导出/导入，支持 `.astrmem` 和 ChromaDB 格式，以及完整备份/恢复。

## 状态：[稳定，本次重构不变动]

功能完备，API 无变化。导出/导入与记忆内部逻辑（衰减、合并、压缩路径）无关。

## 核心类

### MigrationModule

```python
class MigrationModule:
    def __init__(self, data_dir: Path, settings: MemorySettings) -> None: ...

    def export_astrmem(self, user_id: str, output_path: Path) -> dict: ...
    """导出为 .astrmem JSON。返回 {l1_count, l2_count, l3_count, total}。"""

    def import_astrmem(self, user_id: str, input_path: Path) -> dict: ...
    """从 .astrmem JSON 导入。用户ID不匹配时抛 ValueError。"""

    def export_chroma(self, user_id: str, output_dir: Path) -> dict: ...
    """导出 ChromaDB 格式。"""

    def import_chroma(self, user_id: str, input_file: Path) -> None: ...

    def backup(self, backup_dir: Path) -> dict: ...
    """完整备份。返回 {backup_path, l1_count, l2_count}。"""

    def restore(self, backup_path: Path) -> dict: ...
    """从备份恢复。返回 {l1_count, l2_count}。"""
```
