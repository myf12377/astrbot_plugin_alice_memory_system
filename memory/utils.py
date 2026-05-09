"""记忆系统通用工具函数。

P5: Compressor 和 Analyzer 原本各有一份 _parse_score()，逻辑相同只有默认值不同。
提取到本文件共享，通过 default 参数区分:
  Compressor: parse_score(response, default=5)  — 摘要重要性默认为中等
  Analyzer:   parse_score(response, default=0)  — 内容重要性默认为零
"""
import re


def parse_score(response: str, default: int = 0) -> int:
    """从 LLM 回复中提取 0-10 分数。

    LLM 被要求"只输出一个 0-10 的数字"，但有时会附带说明文字。
    用正则提取第一个数字，clamp 到 [0,10]。
    提取失败返回 default 值。
    """
    match = re.search(r"-?\d+", response.strip())
    if match:
        return max(0, min(10, int(match.group())))
    return default
