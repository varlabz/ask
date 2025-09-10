from pydantic_ai.models import Model
from .config import LLMConfig, ProviderEnum

def create_model(llm_config: LLMConfig) -> Model:
    """
    Create a Model from an LLMConfig instance, selecting provider by model prefix.
    """
    model_str = llm_config.model
    if ":" not in model_str:
        raise ValueError(f"Invalid model format: {model_str}. Expected 'provider:model_name' format.")

    provider_name, model_name = model_str.split(":", 1)
    provider_name = provider_name.lower()

    if provider_name == ProviderEnum.GOOGLE.value:
        return _create_google_model(model_name, llm_config)

    if provider_name == ProviderEnum.ANTHROPIC.value:
        return _create_anthropic_model(model_name, llm_config)

    if provider_name in {
        ProviderEnum.OLLAMA.value,
        ProviderEnum.LMSTUDIO.value,
        ProviderEnum.OPENAI.value,
        ProviderEnum.OPENROUTER.value,
    }:
        return _create_openai_compatible_model(provider_name, model_name, llm_config)

    raise ValueError(f"Unsupported provider: {provider_name}")

def _create_google_model(model_name: str, llm_config: LLMConfig) -> Model:
    from pydantic_ai.models.gemini import GeminiModel
    from pydantic_ai.providers.google_gla import GoogleGLAProvider

    provider = GoogleGLAProvider(api_key=llm_config.api_key)
    return GeminiModel(model_name, provider=provider)

def _create_anthropic_model(model_name: str, llm_config: LLMConfig) -> Model:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key=llm_config.api_key)
    return AnthropicModel(model_name, provider=provider)

def _create_openai_compatible_model(provider_name: str, model_name: str, llm_config: LLMConfig) -> Model:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    if provider_name == ProviderEnum.OLLAMA.value:
        provider = OpenAIProvider(base_url=llm_config.base_url or "http://localhost:11434/v1")
    elif provider_name == ProviderEnum.LMSTUDIO.value:
        provider = OpenAIProvider(base_url=llm_config.base_url or "http://localhost:1234/v1")
    elif provider_name == ProviderEnum.OPENAI.value:
        provider = OpenAIProvider(base_url=llm_config.base_url, api_key=llm_config.api_key)
    elif provider_name == ProviderEnum.OPENROUTER.value:
        provider = OpenRouterProvider(api_key=llm_config.api_key or "")
    else:
        raise ValueError(f"Unsupported OpenAI-compatible provider: {provider_name}")

    return OpenAIModel(model_name, provider=provider)



