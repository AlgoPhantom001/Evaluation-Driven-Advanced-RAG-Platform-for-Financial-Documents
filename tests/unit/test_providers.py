from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from secrag.core.errors import AllProvidersFailedError, ProviderError
from secrag.providers.base import ChatMessage, Completion, LLMProvider
from secrag.providers.echo import EchoProvider
from secrag.providers.groq import GroqProvider
from secrag.providers.registry import ProviderChain, build_chain

CONTEXT_PROMPT = """[1] AAPL FY2024 Item 1a Risk Factors
Supply chain disruption could materially reduce revenue. We depend on partners.

[2] AAPL FY2024 Item 7 Mda
Revenue increased two percent during the year. Services grew strongly.

Question: What are the risks?"""


async def test_echo_produces_a_grounded_cited_answer() -> None:
    provider = EchoProvider()
    result = await provider.complete([ChatMessage(role="user", content=CONTEXT_PROMPT)])

    assert "[1]" in result.text
    assert "Supply chain disruption" in result.text
    assert result.prompt_tokens > 0


async def test_echo_is_deterministic() -> None:
    provider = EchoProvider()
    messages = [ChatMessage(role="user", content=CONTEXT_PROMPT)]
    first = await provider.complete(messages)
    second = await provider.complete(messages)
    assert first.text == second.text


async def test_echo_returns_valid_json_in_json_mode() -> None:
    import json

    provider = EchoProvider()
    result = await provider.complete(
        [ChatMessage(role="user", content="Classify the intent of this query")],
        json_mode=True,
    )
    assert isinstance(json.loads(result.text), dict)


async def test_echo_streams_the_same_text() -> None:
    provider = EchoProvider()
    messages = [ChatMessage(role="user", content=CONTEXT_PROMPT)]
    streamed = "".join([piece async for piece in provider.stream(messages)]).strip()
    complete = (await provider.complete(messages)).text.strip()
    assert streamed == complete


@respx.mock
async def test_groq_parses_a_normal_response() -> None:
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama-3.3-70b-versatile",
                "choices": [{"message": {"content": "Revenue rose [1]."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 8},
            },
        )
    )
    provider = GroqProvider(api_key="test", model="llama-3.3-70b-versatile")
    result = await provider.complete([ChatMessage(role="user", content="hi")])

    assert result.text == "Revenue rose [1]."
    assert result.prompt_tokens == 120
    await provider.aclose()


@respx.mock
async def test_groq_raises_on_a_malformed_response() -> None:
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"unexpected": True})
    )
    provider = GroqProvider(api_key="test", model="m", max_retries=0)
    with pytest.raises(ProviderError):
        await provider.complete([ChatMessage(role="user", content="hi")])
    await provider.aclose()


@respx.mock
async def test_groq_does_not_retry_a_client_error() -> None:
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(400, text="bad request")
    )
    provider = GroqProvider(api_key="test", model="m", max_retries=3)
    with pytest.raises(ProviderError):
        await provider.complete([ChatMessage(role="user", content="hi")])
    assert route.call_count == 1
    await provider.aclose()


def test_missing_api_key_fails_fast() -> None:
    with pytest.raises(ProviderError):
        GroqProvider(api_key="", model="m")


class AlwaysFails(LLMProvider):
    name = "broken"

    def __init__(self) -> None:
        super().__init__(model="broken-1")

    async def complete(self, messages, **kwargs) -> Completion:
        msg = "upstream is down"
        raise ProviderError(msg)

    async def stream(self, messages, **kwargs):
        msg = "upstream is down"
        raise ProviderError(msg)
        yield ""


async def test_chain_falls_through_to_a_working_provider() -> None:
    chain = ProviderChain([AlwaysFails(), EchoProvider()])
    result = await chain.complete([ChatMessage(role="user", content=CONTEXT_PROMPT)])
    assert result.provider == "echo"


async def test_chain_raises_only_when_everything_fails() -> None:
    chain = ProviderChain([AlwaysFails(), AlwaysFails()])
    with pytest.raises(AllProvidersFailedError):
        await chain.complete([ChatMessage(role="user", content="hi")])


async def test_chain_streams_through_the_fallback() -> None:
    chain = ProviderChain([AlwaysFails(), EchoProvider()])
    text = "".join(
        [p async for p in chain.stream([ChatMessage(role="user", content=CONTEXT_PROMPT)])]
    )
    assert "[1]" in text


def test_build_chain_degrades_to_offline_when_no_keys_exist(settings) -> None:
    chain = build_chain(settings.model_copy(update={"llm_providers": "groq,gemini"}))
    assert chain.providers[0].name == "echo"


@respx.mock
def test_client_rebinds_when_the_event_loop_changes() -> None:
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": "ok [1]."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )
    )
    provider = GroqProvider(api_key="test", model="m")
    message = [ChatMessage(role="user", content="hi")]

    first = asyncio.run(provider.complete(message))
    client_after_first = provider._client

    second = asyncio.run(provider.complete(message))

    assert first.text == second.text == "ok [1]."
    assert provider._client is not client_after_first, "client was not rebound"
