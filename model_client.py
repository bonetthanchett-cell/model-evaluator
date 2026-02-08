"""
模型客户端 - 支持多种模型提供商的统一接口
"""

import time
from typing import Any, Dict, Optional

import requests


class ModelClient:
    """通用模型客户端"""
    
    def __init__(self, config: Dict[str, Any], timeout: int = 60):
        self.config = config
        self.timeout = timeout
        self.provider = config.get("provider", "openai")
        self.endpoint = config.get("endpoint", "")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.0)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """生成回复"""
        if self.provider in ["openai", "kimi"]:
            return self._generate_openai_compatible(prompt, **kwargs)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai_compatible(self, prompt: str, **kwargs) -> str:
        """OpenAI 兼容格式 API 调用"""
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
    
    def _generate_anthropic(self, prompt: str, **kwargs) -> str:
        """Anthropic Claude API 调用"""
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
