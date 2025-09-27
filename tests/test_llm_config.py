import os

import pytest
from pydantic_ai.models.openai import OpenAIModel

from ask.core.config import LLMConfig
from ask.core.model import create_model


class TestCreateModelFromLLMConfig:
    def test_openai_provider(self):
        llm_config = LLMConfig(
            model="openai:gpt-4o",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        model = create_model(llm_config)
        assert isinstance(model, OpenAIModel)
        assert model.model_name == "gpt-4o"
        assert model.base_url == "https://api.openai.com/v1/"

    def test_ollama_provider(self):
        llm_config = LLMConfig(model="ollama:llama3.2", api_key=None, base_url=None)
        model = create_model(llm_config)
        assert isinstance(model, OpenAIModel)
        assert model.model_name == "llama3.2"
        assert model.base_url == "http://localhost:11434/v1/"

    def test_openrouter_provider(self):
        llm_config = LLMConfig(
            model="openrouter:anthropic/claude-3.5-sonnet",
            api_key="router-key",
            base_url=None,
        )
        model = create_model(llm_config)
        assert isinstance(model, OpenAIModel)
        assert model.model_name == "anthropic/claude-3.5-sonnet"

    def test_invalid_provider(self):
        llm_config = LLMConfig(model="invalid:model", api_key=None, base_url=None)
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            create_model(llm_config)

    def test_invalid_model_format(self):
        llm_config = LLMConfig(model="gpt-4o", api_key=None, base_url=None)
        with pytest.raises(ValueError, match="Invalid model format: gpt-4o"):
            create_model(llm_config)

    def test_api_key_from_file(self):
        """Test that LLMConfig reads api_key from a file."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write("my_secret_key")
            tmp_path = tmp.name
        try:
            config = LLMConfig(model="openai:gpt-4o", api_key=f"file:{tmp_path}")
            assert config.api_key == "my_secret_key"
        finally:
            os.remove(tmp_path)

    def test_api_key_from_file_not_found(self):
        """Test that LLMConfig raises ValueError if api_key file does not exist."""
        fake_path = "/tmp/nonexistent_api_key_file.txt"
        with pytest.raises(ValueError, match=f"File '{fake_path}' not found"):
            LLMConfig(model="openai:gpt-4o", api_key=f"file:{fake_path}")

    def test_google_gemini_model(self, monkeypatch):
        """Test GeminiModel creation with Google provider."""

        class DummyGeminiModel:
            def __init__(self, model_name, provider=None):
                self.model_name = model_name
                self.provider = provider

        class DummyGoogleGLAProvider:
            def __init__(self, api_key=None):
                self.api_key = api_key

        monkeypatch.setattr("pydantic_ai.models.gemini.GeminiModel", DummyGeminiModel)
        monkeypatch.setattr(
            "pydantic_ai.providers.google_gla.GoogleGLAProvider", DummyGoogleGLAProvider
        )

        llm_config = LLMConfig(
            model="google:gemini-2.5-pro", api_key="google-key", base_url=None
        )
        model = create_model(llm_config)
        assert isinstance(model, DummyGeminiModel)
        assert model.model_name == "gemini-2.5-pro"
        assert isinstance(model.provider, DummyGoogleGLAProvider)
        assert model.provider.api_key == "google-key"

    def test_invalid_google_model_format(self):
        llm_config = LLMConfig(
            model="google-gemini-2.5-pro",  # missing colon
            api_key="google-key",
            base_url=None,
        )
        with pytest.raises(
            ValueError, match="Invalid model format: google-gemini-2.5-pro"
        ):
            create_model(llm_config)

    def test_invalid_google_provider(self):
        llm_config = LLMConfig(
            model="googlex:gemini-2.5-pro", api_key="google-key", base_url=None
        )
        with pytest.raises(ValueError, match="Unsupported provider: googlex"):
            create_model(llm_config)
