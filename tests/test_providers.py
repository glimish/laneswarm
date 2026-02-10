"""Tests for the provider registry and protocol."""

import os

import pytest

from laneswarm.providers import (
    LLMResponse,
    Message,
    ProviderConfig,
    ProviderRegistry,
    TokenUsage,
)


def test_provider_config_resolve_api_key():
    config = ProviderConfig(name="test", api_key="sk-123")
    assert config.resolve_api_key() == "sk-123"


def test_provider_config_resolve_api_key_from_env():
    os.environ["TEST_KEY"] = "sk-from-env"
    config = ProviderConfig(name="test", api_key_env="TEST_KEY")
    assert config.resolve_api_key() == "sk-from-env"
    del os.environ["TEST_KEY"]


def test_provider_config_resolve_api_key_none():
    config = ProviderConfig(name="test")
    assert config.resolve_api_key() is None


def test_provider_registry_parse_model_id():
    registry = ProviderRegistry()
    provider, model = registry.parse_model_id("anthropic/claude-opus-4-6")
    assert provider == "anthropic"
    assert model == "claude-opus-4-6"


def test_provider_registry_parse_invalid():
    registry = ProviderRegistry()
    with pytest.raises(ValueError, match="provider/model-name"):
        registry.parse_model_id("invalid-model-id")


def test_provider_registry_get_missing():
    registry = ProviderRegistry()
    with pytest.raises(KeyError, match="not registered"):
        registry.get("nonexistent")


@pytest.mark.asyncio
async def test_provider_registry_complete(mock_registry):
    response = await mock_registry.complete(
        model_id="mock/test-model",
        messages=[Message(role="user", content="hello")],
    )
    assert response.content == "Mock response"
    assert response.usage.input_tokens == 100


def test_token_usage_total():
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    assert usage.total_tokens == 1500
