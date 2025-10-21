from ask.core.config import LLMConfig, load_config

llm = load_config(
    ["~/.config/ask/llm-ollama.yaml"],
    type=LLMConfig,
    key="llm",
)
# use diffetrent LLM config if needed
llm_score = load_config(
    ["~/.config/ask/llm-ollama-no-tools.yaml"], type=LLMConfig, key="llm"
)
