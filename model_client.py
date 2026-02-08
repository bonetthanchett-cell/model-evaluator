"""
模型客户端 - 支持多种模型提供商的统一接口
带重试机制的 API 调用
"""

import time
import logging
from typing import Any, Dict, Optional, Tuple

import requests


class ModelClient:
    """通用模型客户端（带重试机制）"""
    
    def __init__(self, config: Dict[str, Any], timeout: int = 60, max_retries: int = 1):
        self.config = config
        self.timeout = timeout
        self.max_retries = max_retries  # 最大重试次数
        self.provider = config.get("provider", "openai")
        self.endpoint = config.get("endpoint", "")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.0)
        self.logger = logging.getLogger("model_evaluator")
        
    def generate(self, prompt: str, **kwargs) -> Tuple[str, bool, str]:
        """
        生成回复，带重试机制
        
        Returns:
            Tuple[response, success, error_message]
            - response: 模型输出（成功时）或空字符串（失败时）
            - success: 是否成功
            - error_message: 错误信息（失败时）
        """
        if self.provider in ["openai", "kimi"]:
            return self._generate_openai_compatible_with_retry(prompt, **kwargs)
        elif self.provider == "anthropic":
            return self._generate_anthropic_with_retry(prompt, **kwargs)
        else:
            return "", False, f"Unsupported provider: {self.provider}"
    
    def generate_with_messages(self, messages: list, **kwargs) -> Tuple[str, bool, str]:
        """
        使用消息列表生成回复，带重试机制
        
        Returns:
            Tuple[response, success, error_message]
        """
        if self.provider in ["openai", "kimi"]:
            return self._generate_openai_with_messages_retry(messages, **kwargs)
        elif self.provider == "anthropic":
            return self._generate_anthropic_with_messages_retry(messages, **kwargs)
        else:
            return "", False, f"Unsupported provider: {self.provider}"
    
    def _generate_openai_compatible_with_retry(self, prompt: str, **kwargs) -> Tuple[str, bool, str]:
        """OpenAI 兼容格式 API 调用（带重试）"""
        last_error = ""
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self._generate_openai_compatible(prompt, **kwargs)
                return result, True, ""
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    self.logger.warning(f"API 调用失败，准备重试 ({attempt + 1}/{self.max_retries + 1}): {last_error}")
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    self.logger.error(f"API 调用失败，已重试 {self.max_retries} 次仍失败: {last_error}")
        
        return "", False, last_error
    
    def _generate_openai_with_messages_retry(self, messages: list, **kwargs) -> Tuple[str, bool, str]:
        """OpenAI 兼容格式 - 使用消息列表（带重试）"""
        last_error = ""
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self._generate_openai_with_messages(messages, **kwargs)
                return result, True, ""
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    self.logger.warning(f"API 调用失败，准备重试 ({attempt + 1}/{self.max_retries + 1}): {last_error}")
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error(f"API 调用失败，已重试 {self.max_retries} 次仍失败: {last_error}")
        
        return "", False, last_error
    
    def _generate_anthropic_with_retry(self, prompt: str, **kwargs) -> Tuple[str, bool, str]:
        """Anthropic Claude API 调用（带重试）"""
        last_error = ""
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self._generate_anthropic(prompt, **kwargs)
                return result, True, ""
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    self.logger.warning(f"API 调用失败，准备重试 ({attempt + 1}/{self.max_retries + 1}): {last_error}")
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error(f"API 调用失败，已重试 {self.max_retries} 次仍失败: {last_error}")
        
        return "", False, last_error
    
    def _generate_anthropic_with_messages_retry(self, messages: list, **kwargs) -> Tuple[str, bool, str]:
        """Anthropic Claude API - 使用消息列表（带重试）"""
        last_error = ""
        
        for attempt in range(self.max_retries + 1):
            try:
                result = self._generate_anthropic_with_messages(messages, **kwargs)
                return result, True, ""
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    self.logger.warning(f"API 调用失败，准备重试 ({attempt + 1}/{self.max_retries + 1}): {last_error}")
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error(f"API 调用失败，已重试 {self.max_retries} 次仍失败: {last_error}")
        
        return "", False, last_error
    
    def _generate_openai_compatible(self, prompt: str, **kwargs) -> str:
        """OpenAI 兼容格式 API 调用（原始实现）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature)
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        message = data["choices"][0]["message"]
        # Handle both normal content and reasoning field (OpenRouter/Kimi)
        content = message.get("content", "")
        if not content:
            content = message.get("reasoning", "")
        return content
    
    def _generate_openai_with_messages(self, messages: list, **kwargs) -> str:
        """OpenAI 兼容格式 - 使用消息列表（原始实现）"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature)
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        if not content:
            content = message.get("reasoning", "")
        return content
    
    def _generate_anthropic(self, prompt: str, **kwargs) -> str:
        """Anthropic Claude API 调用（原始实现）"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        return data["content"][0]["text"]
    
    def _generate_anthropic_with_messages(self, messages: list, **kwargs) -> str:
        """Anthropic Claude API - 使用消息列表（原始实现）"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": messages
        }
        
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        data = response.json()
        return data["content"][0]["text"]
