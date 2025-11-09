from pydantic import BaseModel
from typing import Dict, Union

class LLM(BaseModel):
    """
    Enhanced LLM base model that supports both predefined cloud models
    and custom local model providers like vLLM and Ollama.
    """
    name: str
    """Name/identifier of the model."""
    max_output_tokens: int
    """Maximum output tokens allowed."""
    temperature: float | None
    """Temperature of the model."""

    # --- New fields for vLLM/Ollama integration ---
    provider: str = "openai"
    """The service providing the model. Defaults to 'openai' for cloud services."""
    api_base: str | None = None
    """The base URL for the API endpoint (e.g., http://localhost:8000)."""
    repetition_penalty: float | None = 1.1
    """The repetition penalty to discourage looping. Defaults to 1.1."""

gemini20flash = LLM(name="gemini-2.0-flash",
                    max_output_tokens=8192,
                    temperature=0.7)
"""`gemini-2.0-flash` model."""

gemini25flash = LLM(name="gemini-2.5-flash",
                    max_output_tokens=65536,
                    temperature=0.7)
"""`gemini-2.5-flash` model."""

gemini25pro = LLM(name="gemini-2.5-pro",
                  max_output_tokens=65536,
                  temperature=0.7)
"""`gemini-2.5-pro` model."""

o3mini = LLM(name="o3-mini-2025-01-31",
             max_output_tokens=100000,
             temperature=None)
"""`o3-mini` model."""

gpt4o = LLM(name="gpt-4o-2024-11-20",
            max_output_tokens=16384,
            temperature=0.5)
"""`gpt-4o` model."""

gpt41 = LLM(name="gpt-4.1-2025-04-14",
            max_output_tokens=16384,
            temperature=0.5)
"""`gpt-4.1` model."""

gpt41mini = LLM(name="gpt-4.1-mini",
                max_output_tokens=16384,
                temperature=0.5)
"""`gpt-4.1-mini` model."""

gpt4omini = LLM(name="gpt-4o-mini-2024-07-18",
                max_output_tokens=16384,
                temperature=0.5)
"""`gpt-4o-mini` model."""

gpt45 = LLM(name="gpt-4.5-preview-2025-02-27",
            max_output_tokens=16384,
            temperature=0.5)
"""`gpt-4.5-preview` model."""

gpt5 = LLM(name="gpt-5",
           max_output_tokens=128000,
           temperature=None)
"""`gpt-5` model """

gpt5mini = LLM(name="gpt-5-mini",
               max_output_tokens=128000,
               temperature=None)
"""`gpt-5-mini` model."""

claude37sonnet = LLM(name="claude-3-7-sonnet-20250219",
                     max_output_tokens=64000,
                     temperature=0)
"""`claude-3-7-sonnet` model."""

claude4opus = LLM(name="claude-opus-4-20250514",
                   max_output_tokens=32000,
                   temperature=0)
"""`claude-4-Opus` model."""

claude41opus = LLM(name="claude-opus-4-1-20250805",
                   max_output_tokens=32000,
                   temperature=0)
"""`claude-4.1-Opus` model."""

models : Dict[str, LLM] = {
                            "gemini-2.0-flash" : gemini20flash,
                            "gemini-2.5-flash" : gemini25flash,
                            "gemini-2.5-pro" : gemini25pro,
                            "o3-mini" : o3mini,
                            "gpt-4o" : gpt4o,
                            "gpt-4.1" : gpt41,
                            "gpt-4.1-mini" : gpt41mini,
                            "gpt-4o-mini" : gpt4omini,
                            "gpt-4.5" : gpt45,
                            "gpt-5" : gpt5,
                            "gpt-5-mini" : gpt5mini,
                            "claude-3.7-sonnet" : claude37sonnet,
                            "claude-4-opus" : claude4opus,
                            "claude-4.1-opus" : claude41opus,
                           }
"""Dictionary with the available models."""



# --- NEW: The universal LLM parser function ---
def llm_parser(model_config: Union[str, Dict, LLM]) -> LLM:
    """
    Parses a model identifier and returns a configured LLM instance.

    This function is the bridge between the Streamlit UI and the Denario backend.

    Args:
        model_config: Can be one of three types:
            - str: A key to look up a predefined model in the `models` dict (e.g., "gpt-4o").
            - LLM: If an LLM object is passed, it's returned directly.
            - dict: A detailed configuration from the Streamlit UI, used to create
                    a new LLM instance for vLLM or Ollama on-the-fly.

    Returns:
        An instance of the LLM class.
    """
    if isinstance(model_config, LLM):
        # If it's already a valid LLM object, just return it.
        return model_config

    if isinstance(model_config, str):
        # For backward compatibility, look up the string in the predefined models.
        if model_config in models:
            return models[model_config]
        else:
            raise ValueError(f"Model string '{model_config}' not found in predefined models.")

    if isinstance(model_config, dict):
        # This is the new, flexible path for vLLM/Ollama from the Streamlit UI.
        # Create an LLM instance directly from the dictionary.
        return LLM(
            name=model_config['model'],
            provider=model_config['provider'],
            api_base=model_config['api_base'],
            temperature=model_config.get('temperature'),
            max_output_tokens=model_config.get('max_tokens'),
            repetition_penalty=model_config.get('repetition_penalty')
        )
    
    raise TypeError(f"Unsupported model configuration type: {type(model_config)}")
