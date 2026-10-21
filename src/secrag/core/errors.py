from __future__ import annotations


class SecRagError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class ConfigurationError(SecRagError):
    status_code = 500
    code = "configuration_error"


class ProviderError(SecRagError):
    status_code = 502
    code = "provider_error"


class AllProvidersFailedError(ProviderError):
    code = "all_providers_failed"


class RateLimitError(ProviderError):
    status_code = 429
    code = "rate_limited"


class IngestionError(SecRagError):
    status_code = 422
    code = "ingestion_error"


class EdgarError(IngestionError):
    code = "edgar_error"


class RetrievalError(SecRagError):
    status_code = 500
    code = "retrieval_error"


class IndexNotReadyError(RetrievalError):
    status_code = 409
    code = "index_not_ready"


class GuardrailTrippedError(SecRagError):
    status_code = 200
    code = "guardrail_tripped"


class EvaluationError(SecRagError):
    status_code = 500
    code = "evaluation_error"
