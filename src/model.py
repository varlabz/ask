from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from config import LLMConfig, ProviderEnum


def create_model(llm_config: LLMConfig) -> OpenAIModel:
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

    provider = None
    if provider_name == ProviderEnum.OLLAMA.value:
        provider = OpenAIProvider(
            base_url=llm_config.base_url or "http://localhost:11434/v1",
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


def create_model_from_llm_config(llm_config: LLMConfig) -> OpenAIModel:
    """Alias for create_model function for backward compatibility.
    
    Args:
        llm_config: LLMConfig object containing model and provider settings.
    
    Returns:
        OpenAIModel: Configured OpenAIModel instance.
    """
    return create_model(llm_config)
