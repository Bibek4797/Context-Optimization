from __future__ import annotations

import httpx
from app.models.schemas import CountType, ModelInfo, TokenMeasurement
from app.services.llm.base import LLMConfigurationError, LLMResponse, LLMProvider
from app.services.token_service import TokenService


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        import os
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.provider = "groq"
        self._token_service = TokenService()

    def count_tokens(self, text: str, stage: str = "llm_prompt_tokens") -> TokenMeasurement:
        estimate = self._token_service.estimate_tokens(text)
        return TokenMeasurement(
            stage=stage,
            tokens=estimate,
            count_type=CountType.estimated,
            provider=self.provider,
            model=self.model,
        )

    def generate_answer(self, prompt: str) -> LLMResponse:
        if not self.api_key:
            raise LLMConfigurationError("Groq API Key is not configured.")
            
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=data)
                
            if response.status_code != 200:
                raise RuntimeError(f"Groq API error ({response.status_code}): {response.text}")
                
            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                raise RuntimeError("Empty response from Groq API.")
                
            text = choices[0].get("message", {}).get("content", "")
            
            usage = res_json.get("usage")
            if usage:
                prompt_count = usage.get("prompt_tokens", 0)
                response_count = usage.get("completion_tokens", 0)
                total_count = usage.get("total_tokens", 0)
                
                prompt_tokens = TokenMeasurement(
                    stage="llm_prompt_tokens",
                    tokens=prompt_count,
                    count_type=CountType.exact,
                    provider=self.provider,
                    model=self.model,
                )
                response_tokens = TokenMeasurement(
                    stage="llm_response_tokens",
                    tokens=response_count,
                    count_type=CountType.exact,
                    provider=self.provider,
                    model=self.model,
                )
                total_tokens = TokenMeasurement(
                    stage="total_per_query_tokens",
                    tokens=total_count,
                    count_type=CountType.exact,
                    provider=self.provider,
                    model=self.model,
                )
            else:
                prompt_tokens = self.count_tokens(prompt, "llm_prompt_tokens")
                response_tokens = self._token_service.measure_estimated("llm_response_tokens", text)
                total_tokens = TokenMeasurement(
                    stage="total_per_query_tokens",
                    tokens=prompt_tokens.tokens + response_tokens.tokens,
                    count_type=CountType.estimated,
                    provider=self.provider,
                    model=self.model,
                )
                
            return LLMResponse(text=text, prompt_tokens=prompt_tokens, response_tokens=response_tokens, total_tokens=total_tokens)
        except Exception as exc:
            raise RuntimeError(f"Groq generation failed: {exc}") from exc

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.provider,
            model=self.model,
            configured=bool(self.api_key),
            notes=None if self.api_key else "Set Groq API Key to enable calls.",
        )


class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        import os
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.provider = "openrouter"
        self._token_service = TokenService()

    def count_tokens(self, text: str, stage: str = "llm_prompt_tokens") -> TokenMeasurement:
        estimate = self._token_service.estimate_tokens(text)
        return TokenMeasurement(
            stage=stage,
            tokens=estimate,
            count_type=CountType.estimated,
            provider=self.provider,
            model=self.model,
        )

    def generate_answer(self, prompt: str) -> LLMResponse:
        if not self.api_key:
            raise LLMConfigurationError("OpenRouter API Key is not configured.")
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Context Optimization Engine",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=data)
                
            if response.status_code != 200:
                raise RuntimeError(f"OpenRouter API error ({response.status_code}): {response.text}")
                
            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                raise RuntimeError("Empty response from OpenRouter API.")
                
            text = choices[0].get("message", {}).get("content", "")
            
            usage = res_json.get("usage")
            if usage:
                prompt_count = usage.get("prompt_tokens", 0)
                response_count = usage.get("completion_tokens", 0)
                total_count = usage.get("total_tokens", 0)
                
                prompt_tokens = TokenMeasurement(
                    stage="llm_prompt_tokens",
                    tokens=prompt_count,
                    count_type=CountType.exact,
                    provider=self.provider,
                    model=self.model,
                )
                response_tokens = TokenMeasurement(
                    stage="llm_response_tokens",
                    tokens=response_count,
                    count_type=CountType.exact,
                    provider=self.provider,
                    model=self.model,
                )
                total_tokens = TokenMeasurement(
                    stage="total_per_query_tokens",
                    tokens=total_count,
                    count_type=CountType.exact,
                    provider=self.provider,
                    model=self.model,
                )
            else:
                prompt_tokens = self.count_tokens(prompt, "llm_prompt_tokens")
                response_tokens = self._token_service.measure_estimated("llm_response_tokens", text)
                total_tokens = TokenMeasurement(
                    stage="total_per_query_tokens",
                    tokens=prompt_tokens.tokens + response_tokens.tokens,
                    count_type=CountType.estimated,
                    provider=self.provider,
                    model=self.model,
                )
                
            return LLMResponse(text=text, prompt_tokens=prompt_tokens, response_tokens=response_tokens, total_tokens=total_tokens)
        except Exception as exc:
            raise RuntimeError(f"OpenRouter generation failed: {exc}") from exc

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.provider,
            model=self.model,
            configured=bool(self.api_key),
            notes=None if self.api_key else "Set OpenRouter API Key to enable calls.",
        )
