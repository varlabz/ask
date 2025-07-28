from pydantic_ai.models import Model

from config import LLMConfig, ProviderEnum


def create_model(llm_config: LLMConfig) -> Model:
    """Create an OpenAIModel from an LLMConfig instance, selecting provider by model prefix.
    
    Args:
        llm_config: LLMConfig object containing model and provider settings.
    
    Returns:
        OpenAIModel: Configured OpenAIModel instance.
    """
    model_str = llm_config.model
    if ":" in model_str:
        provider_name, model_name = model_str.split(":", 1)
    else:
        raise ValueError(f"Invalid model format: {model_str}. Expected 'provider:model_name' format.")

    provider_name = provider_name.lower()
    if provider_name == ProviderEnum.GOOGLE.value:
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.providers.google_gla import GoogleGLAProvider
        provider = GoogleGLAProvider(
            api_key=llm_config.api_key,
        )
        return GeminiModel(
            model_name,
            provider=provider,
        )

    # openai compatible models
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    provider = None
    if provider_name == ProviderEnum.OLLAMA.value:
        provider = OpenAIProvider(
            base_url=llm_config.base_url or "http://localhost:11434/v1",
        )
    elif provider_name == ProviderEnum.LMSTUDIO.value:
        provider = OpenAIProvider(
            base_url=llm_config.base_url or "http://localhost:1234/v1",
        )
    elif provider_name == ProviderEnum.OPENAI.value:
        provider = OpenAIProvider(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
        )
    elif provider_name == ProviderEnum.OPENROUTER.value:
        provider = OpenRouterProvider(
            api_key=llm_config.api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

    model = OpenAIModel(
        model_name,
        provider=provider,
    )
    return model


