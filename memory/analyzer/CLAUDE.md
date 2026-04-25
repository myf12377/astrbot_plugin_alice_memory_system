# Analyzer Module

## 角色

LLM 重要性分析：单条打分、灰区批量重评、L3 记忆合并。

## 状态：重构中

## 构造

```python
def __init__(self, context: Any, config: PluginConfig) -> None
```

## 公开 API

```python
async def analyze(self, content: str) -> int: ...
"""单条分析，返回 0-10 分数。"""

def should_promote_to_l3(self, content: str) -> bool: ...
"""分数 ≥ importance_threshold → True。"""

async def batch_recheck(self, memories: list[dict]) -> list[dict]: ...
"""灰区批量重评。
输入: [{content, metadata, ...}, ...]
输出: [{vector_id, new_score, should_keep}, ...]"""

async def merge_content(self, content_1: str, content_2: str) -> str: ...
"""LLM 合并两条相似记忆，去冗余保留关键信息。"""
```

内部：`_build_prompt`, `_build_batch_prompt`, `_build_merge_prompt`, `_parse_score`。

## 边界

不负责：记忆 CRUD（由调用方执行）、压缩、定时调度。
依赖方：Main, Scheduler, VectorStore(merge_content)。
