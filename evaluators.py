"""
评估器模块 - 支持多种评估指标
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseEvaluator(ABC):
    """评估器基类"""
    
    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """评估预测结果"""
        pass


class ExactMatchEvaluator(BaseEvaluator):
    """精确匹配评估器"""
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        expected = ground_truth.get("output", {}).get("answer", "")
        
        # 标准化：去除多余空白
        pred_norm = " ".join(prediction.strip().split())
        expected_norm = " ".join(str(expected).strip().split())
        
        correct = pred_norm.lower() == expected_norm.lower()
        
        return {
            "correct": correct,
            "metric": "exact_match",
            "expected": expected,
            "predicted": prediction
        }


class NumberMatchEvaluator(BaseEvaluator):
    """数字匹配评估器（从文本中提取数字比较）"""
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        expected = ground_truth.get("output", {}).get("answer", "")
        
        # 从预测中提取数字
        pred_numbers = self._extract_numbers(prediction)
        expected_number = self._extract_single_number(str(expected))
        
        if expected_number is not None and pred_numbers:
            # 取最后一个数字（通常是答案）
            pred_number = pred_numbers[-1]
            correct = abs(pred_number - expected_number) < 1e-6
        else:
            correct = False
            pred_number = None
        
        return {
            "correct": correct,
            "metric": "number_match",
            "expected": expected_number,
            "predicted": pred_number,
            "extracted_numbers": pred_numbers
        }
    
    def _extract_numbers(self, text: str) -> list:
        """从文本中提取所有数字"""
        # 支持整数、小数、负数
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(m) if '.' in m else int(m) for m in matches]
    
    def _extract_single_number(self, text: str) -> Optional[float]:
        """从文本中提取单个数字"""
        numbers = self._extract_numbers(text)
        return numbers[0] if numbers else None


class ContainsMatchEvaluator(BaseEvaluator):
    """包含匹配评估器（预测包含正确答案）"""
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        expected = str(ground_truth.get("output", {}).get("answer", ""))
        
        correct = expected.lower() in prediction.lower()
        
        return {
            "correct": correct,
            "metric": "contains_match",
            "expected": expected,
            "predicted": prediction
        }


class CodeExecutionEvaluator(BaseEvaluator):
    """代码执行评估器（检查代码是否通过测试用例）"""
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        tests = ground_truth.get("output", {}).get("tests", [])
        
        # 这里简化处理，实际应该执行代码
        # TODO: 实现安全的代码沙箱执行
        
        return {
            "correct": False,  # 暂不实现
            "metric": "code_execution",
            "note": "Code execution not implemented yet",
            "tests_count": len(tests)
        }


def get_evaluator(task: Dict[str, Any]) -> BaseEvaluator:
    """根据任务获取对应的评估器"""
    eval_config = task.get("evaluation", {})
    metric = eval_config.get("metric", "exact_match")
    checker = eval_config.get("checker", "")
    
    # 根据 checker 优先选择
    if checker == "extract_and_compare_number":
        return NumberMatchEvaluator()
    
    # 根据 metric 选择
    evaluators = {
        "exact_match": ExactMatchEvaluator(),
        "accuracy": ExactMatchEvaluator(),
        "contains": ContainsMatchEvaluator(),
        "pass@k": CodeExecutionEvaluator(),
    }
    
    return evaluators.get(metric, ExactMatchEvaluator())
