"""
评估器模块 - 支持多种评估指标
"""

import re
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests


class BaseEvaluator(ABC):
    """评估器基类"""
    
    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """评估预测结果"""
        pass


class ExactMatchEvaluator(BaseEvaluator):
    """精确匹配评估器"""
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        # 处理 None 或空字符串
        if prediction is None:
            prediction = ""
        
        expected = ground_truth.get("output", {}).get("answer", "")
        
        # 标准化：去除多余空白
        pred_norm = " ".join(str(prediction).strip().split())
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
        # 处理 None 或空字符串
        if prediction is None:
            prediction = ""
        
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
        if not text:
            return []
        # 支持整数、小数、负数
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(m) if '.' in m else int(m) for m in matches]
    
    def _extract_single_number(self, text: str) -> Optional[float]:
        """从文本中提取单个数字"""
        if not text:
            return None
        numbers = self._extract_numbers(text)
        return numbers[0] if numbers else None


class ContainsMatchEvaluator(BaseEvaluator):
    """包含匹配评估器（预测包含正确答案）"""
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        # 处理 None 或空字符串
        if prediction is None:
            prediction = ""
        
        expected = str(ground_truth.get("output", {}).get("answer", ""))
        
        # 处理空值情况
        if not expected:
            return {
                "correct": False,
                "metric": "contains_match",
                "expected": expected,
                "predicted": prediction
            }
        
        correct = expected.lower() in str(prediction).lower()
        
        return {
            "correct": correct,
            "metric": "contains_match",
            "expected": expected,
            "predicted": prediction
        }


class CodeExecutionEvaluator(BaseEvaluator):
    """代码执行评估器（检查代码是否通过测试用例）"""
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        # 处理 None 或空字符串
        if prediction is None:
            prediction = ""
            
        tests = ground_truth.get("output", {}).get("tests", [])
        
        # 这里简化处理，实际应该执行代码
        # TODO: 实现安全的代码沙箱执行
        
        return {
            "correct": False,  # 暂不实现
            "metric": "code_execution",
            "note": "Code execution not implemented yet",
            "tests_count": len(tests)
        }


class LLMJudgeEvaluator(BaseEvaluator):
    """LLM Judge 评估器 - 使用大模型判断答案是否一致"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 LLM Judge 评估器
        
        Args:
            config: 配置字典，包含:
                - endpoint: API 端点
                - model: 模型名称
                - api_key_env: API key 环境变量名
                - system_prompt: System prompt
                - user_template: User prompt 模板
                - max_tokens: 最大 token 数
                - temperature: 温度
                - timeout: 超时时间
        """
        self.config = config or {}
        self.endpoint = self.config.get("endpoint", "https://api.moonshot.cn/v1/chat/completions")
        self.model = self.config.get("model", "kimi-k2.5")
        self.api_key_env = self.config.get("api_key_env", "MOONSHOT_API_KEY")
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.0)
        self.timeout = self.config.get("timeout", 30)
        
        # 默认 system prompt
        self.system_prompt = self.config.get(
            "system_prompt",
            "You are an evaluator. Determine if the predicted answer matches the expected answer.\n"
            "Respond with only 'yes' if they match, or 'no' if they don't match."
        )
        
        # 默认 user template
        self.user_template = self.config.get(
            "user_template",
            "# {prediction}\n\n# {ground_truth}"
        )
        
        self.api_key = os.getenv(self.api_key_env)
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 LLM 评估预测结果
        
        Args:
            prediction: 模型预测结果
            ground_truth: 包含期望答案的字典
            
        Returns:
            评估结果字典
        """
        # 处理 None 或空字符串
        if prediction is None:
            prediction = ""
        
        expected = str(ground_truth.get("output", {}).get("answer", ""))
        
        # 如果 API key 未设置，返回失败
        if not self.api_key:
            return {
                "correct": False,
                "metric": "llm_judge",
                "expected": expected,
                "predicted": prediction,
                "error": f"API key not found in environment: {self.api_key_env}",
                "llm_response": None
            }
        
        try:
            # 构建 user prompt
            user_prompt = self.user_template.format(
                prediction=prediction,
                ground_truth=expected
            )
            
            # 调用 LLM API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            llm_response = data["choices"][0]["message"]["content"].strip().lower()
            
            # 解析响应 - 检查是否包含 yes
            # 支持 "yes", "yes.", "yes,", "(yes)", "**yes**" 等格式
            is_correct = "yes" in llm_response and "no" not in llm_response
            
            return {
                "correct": is_correct,
                "metric": "llm_judge",
                "expected": expected,
                "predicted": prediction,
                "llm_response": llm_response,
                "user_prompt": user_prompt
            }
            
        except Exception as e:
            return {
                "correct": False,
                "metric": "llm_judge",
                "expected": expected,
                "predicted": prediction,
                "error": str(e),
                "llm_response": None
            }


def get_evaluator(task: Dict[str, Any], llm_judge_config: Optional[Dict[str, Any]] = None) -> BaseEvaluator:
    """
    根据任务获取对应的评估器
    
    Args:
        task: 任务字典
        llm_judge_config: LLM Judge 的配置（可选）
    """
    eval_config = task.get("evaluation", {})
    metric = eval_config.get("metric", "exact_match")
    checker = eval_config.get("checker", "")
    
    # 根据 checker 优先选择
    if checker == "extract_and_compare_number":
        return NumberMatchEvaluator()
    
    # 根据 metric 选择
    if metric == "llm_judge":
        return LLMJudgeEvaluator(llm_judge_config)
    
    evaluators = {
        "exact_match": ExactMatchEvaluator(),
        "accuracy": ExactMatchEvaluator(),
        "contains": ContainsMatchEvaluator(),
        "pass@k": CodeExecutionEvaluator(),
    }
    
    return evaluators.get(metric, ExactMatchEvaluator())
