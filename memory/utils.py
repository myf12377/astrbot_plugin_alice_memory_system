"""记忆系统通用工具函数。"""
import re


def parse_score(response: str, default: int = 0) -> int:
    """从 LLM 回复中提取 0-10 分数。"""
    match = re.search(r"-?\d+", response.strip())
    if match:
        return max(0, min(10, int(match.group())))
    return default
