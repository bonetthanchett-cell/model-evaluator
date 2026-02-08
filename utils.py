"""
工具函数模块
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    data = []
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data


def save_jsonl(filepath: str, data: List[Dict[str, Any]]):
    """保存为 JSONL 文件"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_result(result: Dict[str, Any], verbose: bool = False) -> str:
    """格式化评估结果用于显示"""
    task_id = result.get("task_id", "unknown")
    success = result.get("success", False)
    evaluation = result.get("evaluation", {})
    correct = evaluation.get("correct", False)
    
    status = "✓" if correct else "✗"
    if not success:
        status = "⚠"
    
    output = f"{status} {task_id}"
    
    if verbose:
        output += f"\n  Prediction: {result.get('prediction', 'N/A')[:100]}..."
        output += f"\n  Expected: {evaluation.get('expected', 'N/A')}"
        if not success:
            output += f"\n  Error: {result.get('error', 'Unknown')}"
    
    return output


def truncate_text(text: str, max_length: int = 100) -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
