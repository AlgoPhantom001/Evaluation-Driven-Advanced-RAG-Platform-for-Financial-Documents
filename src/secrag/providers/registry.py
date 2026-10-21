from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from secrag.core.config import Settings, get_settings
from secrag.core.errors import AllProvidersFailedError, ConfigurationError, ProviderError
from secrag.core.logging import get_logger
from secrag.providers.base import ChatMessage, Completion, HTTPProvider, LLMProvider
from secrag.providers.echo import EchoProvider
from secrag.providers.gemini import GeminiProvider
from secrag.providers.groq import GroqProvider

log = get_logger(__name__)

_BUILDERS: dict[str, type[HTTPProvider]] = {"groq": GroqProvider, "gemini": GeminiProvider}


def build_provider(name: str, settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    key = name.strip().lower()
    if key == "echo":
        return EchoProvider(model=settings.model_for(key))

    builder = _BUILDERS.get(key)
    if builder is None:
        valid = sorted([*_BUILDERS, "echo"])
        msg = f"Unknown provider {name!r}. Valid options: {valid}"
        raise ConfigurationError(msg)

    provider: LLMProvider = builder(
        api_key=settings.api_key_for(key),
        model=settings.model_for(key),
        timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
    )
    return provider


class ProviderChain(LLMProvider):
    name = "chain"

    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        if not providers:
            msg = "ProviderChain requires at least one provider"
            raise ConfigurationError(msg)
        super().__init__(model=providers[0].model)
        self.providers = list(providers)

    @property
    def primary(self) -> LLMProvider:
        return self.providers[0]

    def describe(self) -> list[str]:
        return [f"{p.name}:{p.model}" for p in self.providers]

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> Completion:
        failures: list[str] = []
        for provider in self.providers:
            try:
                return await provider.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            except ProviderError as exc:
                failures.append(f"{provider.name}: {exc.message}")
                log.warning("provider_failed", provider=provider.name, error=exc.message)
            except Exception as exc:
                failures.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                log.warning("provider_errored", provider=provider.name, error=str(exc))

        msg = "Every configured LLM provider failed"
        raise AllProvidersFailedError(msg, detail=failures)

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        failures: list[str] = []
        for provider in self.providers:
            produced = False
            try:
                async for piece in provider.stream(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    produced = True
                    yield piece
            except Exception as exc:
                failures.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                log.warning("provider_stream_failed", provider=provider.name, error=str(exc))
                if produced:
                    raise
                continue
            else:
                return

        msg = "Every configured LLM provider failed while streaming"
        raise AllProvidersFailedError(msg, detail=failures)

    async def aclose(self) -> None:
        for provider in self.providers:
            if (closer := getattr(provider, "aclose", None)) is not None:
                await closer()


def build_chain(settings: Settings | None = None) -> ProviderChain:
    settings = settings or get_settings()
    names = settings.configured_providers()

    if not names:
        log.warning("no_providers_configured", detail="falling back to offline echo provider")
        names = ["echo"]

    providers: list[LLMProvider] = []
    for name in names:
        try:
            providers.append(build_provider(name, settings))
        except (ProviderError, ConfigurationError) as exc:
            log.warning("provider_build_failed", provider=name, error=str(exc))

    if not providers:
        providers.append(EchoProvider())

    chain = ProviderChain(providers)
    log.info("provider_chain_ready", chain=chain.describe())
    return chain
