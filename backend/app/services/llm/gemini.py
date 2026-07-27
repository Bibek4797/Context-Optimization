from __future__ import annotations

from app.models.schemas import CountType, ModelInfo, TokenMeasurement
from app.services.llm.base import LLMConfigurationError, LLMResponse, LLMProvider
from app.services.token_service import TokenService


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.provider = "gemini"
        self._client = None
        self._token_service = TokenService()

    @property
    def client(self):
        if not self.api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is not configured.")
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - dependency issue
                raise LLMConfigurationError("google-genai is not installed.") from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def count_tokens(self, text: str, stage: str = "llm_prompt_tokens") -> TokenMeasurement:
        try:
            response = self.client.models.count_tokens(model=self.model, contents=text)
            total = int(getattr(response, "total_tokens", 0) or 0)
            return TokenMeasurement(
                stage=stage,
                tokens=total,
                count_type=CountType.exact,
                provider=self.provider,
                model=self.model,
            )
        except LLMConfigurationError:
            raise
        except Exception as exc:
            estimate = self._token_service.estimate_tokens(text)
            return TokenMeasurement(
                stage=stage,
                tokens=estimate,
                count_type=CountType.estimated,
                provider=self.provider,
                model=self.model,
                notes=f"Gemini token count failed: {exc}",
            )

    def generate_answer(self, prompt: str) -> LLMResponse:
        prompt_tokens = self.count_tokens(prompt, "llm_prompt_tokens")
        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt)
        except LLMConfigurationError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc

        text = getattr(response, "text", "") or ""
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_count = int(getattr(usage, "prompt_token_count", 0) or prompt_tokens.tokens)
            response_count = int(getattr(usage, "candidates_token_count", 0) or 0)
            total_count = int(getattr(usage, "total_token_count", 0) or (prompt_count + response_count))
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
            response_tokens = self._token_service.measure_estimated("llm_response_tokens", text)
            total_tokens = TokenMeasurement(
                stage="total_per_query_tokens",
                tokens=prompt_tokens.tokens + response_tokens.tokens,
                count_type=CountType.estimated,
                provider=self.provider,
                model=self.model,
                notes="Gemini usage metadata missing; response tokens estimated.",
            )
        return LLMResponse(text=text, prompt_tokens=prompt_tokens, response_tokens=response_tokens, total_tokens=total_tokens)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.provider,
            model=self.model,
            configured=bool(self.api_key),
            notes=None if self.api_key else "Set GEMINI_API_KEY in .env to enable LLM calls.",
        )

