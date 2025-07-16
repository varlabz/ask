import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from config import LLMConfig, ProviderEnum
from model import create_model_from_llm_config
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

class TestCreateModelFromLLMConfig:
    def test_openai_provider(self):
        llm_config = LLMConfig(
            model="openai:gpt-4o",
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )
        model = create_model_from_llm_config(llm_config)
        assert isinstance(model, OpenAIModel)
        assert model.model_name == "gpt-4o"
        assert model.base_url == "https://api.openai.com/v1/"

    def test_ollama_provider(self):
        llm_config = LLMConfig(
            model="ollama:llama3.2",
            api_key=None,
            base_url=None
        )
        model = create_model_from_llm_config(llm_config)
        assert isinstance(model, OpenAIModel)
        assert model.model_name == "llama3.2"
        assert model.base_url == "http://localhost:11434/v1/"

    def test_openrouter_provider(self):
        from pydantic_ai.providers.openrouter import OpenRouterProvider
        llm_config = LLMConfig(
            model="openrouter:anthropic/claude-3.5-sonnet",
            api_key="router-key",
            base_url=None
        )
        model = create_model_from_llm_config(llm_config)
        assert isinstance(model, OpenAIModel)
        assert model.model_name == "anthropic/claude-3.5-sonnet"

    def test_invalid_provider(self):
        llm_config = LLMConfig(
            model="invalid:model",
            api_key=None,
            base_url=None
        )
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            create_model_from_llm_config(llm_config)

    def test_invalid_model_format(self):
        llm_config = LLMConfig(
            model="gpt-4o",
            api_key=None,
            base_url=None
        )
        with pytest.raises(ValueError, match="Invalid model format: gpt-4o"):
            create_model_from_llm_config(llm_config)
