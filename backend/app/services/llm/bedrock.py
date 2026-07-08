from __future__ import annotations

from app.models.schemas import CountType, ModelInfo, TokenMeasurement
from app.services.llm.base import LLMConfigurationError, LLMResponse, LLMProvider
from app.services.token_service import TokenService


class BedrockProvider(LLMProvider):
    def __init__(
        self,
        aws_access_key_id: str | None,
        aws_secret_access_key: str | None,
        aws_region: str | None,
        model: str
    ) -> None:
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region or "us-east-1"
        self.model = model
        self.provider = "bedrock"
        self._client = None
        self._token_service = TokenService()

    @property
    def client(self):
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise LLMConfigurationError("AWS Access Key ID and Secret Access Key must be configured for Bedrock.")
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise LLMConfigurationError("boto3 is not installed. Please install it using pip.") from exc
            
            try:
                self._client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self.aws_region,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                )
            except Exception as exc:
                raise LLMConfigurationError(f"Failed to create Bedrock client: {exc}") from exc
        return self._client

    def count_tokens(self, text: str, stage: str = "llm_prompt_tokens") -> TokenMeasurement:
        # Bedrock doesn't expose a lightweight token counter API for arbitrary text,
        # so we fallback to local token estimation.
        estimate = self._token_service.estimate_tokens(text)
        return TokenMeasurement(
            stage=stage,
            tokens=estimate,
            count_type=CountType.estimated,
            provider=self.provider,
            model=self.model,
        )

    def generate_answer(self, prompt: str) -> LLMResponse:
        try:
            client = self.client
            
            response = client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": prompt}]}]
            )
            
            # Extract content text
            output_msg = response.get("output", {}).get("message", {})
            content = output_msg.get("content", [])
            text = ""
            if content and "text" in content[0]:
                text = content[0]["text"]
            
            # Extract usage metadata
            usage = response.get("usage", {})
            prompt_count = usage.get("inputTokens", 0)
            response_count = usage.get("outputTokens", 0)
            total_count = usage.get("totalTokens", 0)
            
            prompt_tokens = TokenMeasurement(
                stage="llm_prompt_tokens",
                tokens=prompt_count,
                count_type=CountType.exact if prompt_count > 0 else CountType.estimated,
                provider=self.provider,
                model=self.model,
            )
            response_tokens = TokenMeasurement(
                stage="llm_response_tokens",
                tokens=response_count,
                count_type=CountType.exact if response_count > 0 else CountType.estimated,
                provider=self.provider,
                model=self.model,
            )
            total_tokens = TokenMeasurement(
                stage="total_per_query_tokens",
                tokens=total_count,
                count_type=CountType.exact if total_count > 0 else CountType.estimated,
                provider=self.provider,
                model=self.model,
            )
            
            return LLMResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens
            )
        except LLMConfigurationError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Bedrock generation failed: {exc}") from exc

    def get_model_info(self) -> ModelInfo:
        configured = bool(self.aws_access_key_id and self.aws_secret_access_key)
        return ModelInfo(
            provider=self.provider,
            model=self.model,
            configured=configured,
            notes=None if configured else "AWS Access/Secret Keys must be configured for Bedrock.",
        )
