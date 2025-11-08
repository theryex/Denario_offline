from typing import Dict, Any
from langchain_openai import ChatOpenAI

class LLM(ChatOpenAI):
    """
    LLM base model that extends langchain's ChatOpenAI model to include additional metadata.
    """
    # model_name, max_tokens, and temperature are inherited from ChatOpenAI.
    model_type: str = "remote"
    """Type of the model, can be 'remote' or 'local'."""
    client: str | None = None
    """Client to be used for local models, e.g., 'vllm' or 'ollama'."""

    @property
    def name(self) -> str:
        """Alias for model_name for backward compatibility."""
        return self.model_name

    @property
    def max_output_tokens(self) -> int:
        """Alias for max_tokens for backward compatibility."""
        # The parent class might not have max_tokens set, so provide a default.
        return self.max_tokens or 0

gemini20flash = LLM(model_name="gemini-2.0-flash",
                    max_tokens=8192,
                    temperature=0.7)
"""`gemini-2.0-flash` model."""

gemini25flash = LLM(model_name="gemini-2.5-flash",
                    max_tokens=65536,
                    temperature=0.7)
"""`gemini-2.5-flash` model."""

gemini25pro = LLM(model_name="gemini-2.5-pro",
                  max_tokens=65536,
                  temperature=0.7)
"""`gemini-2.5-pro` model."""

o3mini = LLM(model_name="o3-mini-2025-01-31",
             max_tokens=100000,
             temperature=None)
"""`o3-mini` model."""

gpt4o = LLM(model_name="gpt-4o-2024-11-20",
            max_tokens=16384,
            temperature=0.5)
"""`gpt-4o` model."""

gpt41 = LLM(model_name="gpt-4.1-2025-04-14",
            max_tokens=16384,
            temperature=0.5)
"""`gpt-4.1` model."""

gpt41mini = LLM(model_name="gpt-4.1-mini",
                max_tokens=16384,
                temperature=0.5)
"""`gpt-4.1-mini` model."""

gpt4omini = LLM(model_name="gpt-4o-mini-2024-07-18",
                max_tokens=16384,
                temperature=0.5)
"""`gpt-4o-mini` model."""

gpt45 = LLM(model_name="gpt-4.5-preview-2025-02-27",
            max_tokens=16384,
            temperature=0.5)
"""`gpt-4.5-preview` model."""

gpt5 = LLM(model_name="gpt-5",
           max_tokens=128000,
           temperature=None)
"""`gpt-5` model """

gpt5mini = LLM(model_name="gpt-5-mini",
               max_tokens=128000,
               temperature=None)
"""`gpt-5-mini` model."""

claude37sonnet = LLM(model_name="claude-3-7-sonnet-20250219",
                     max_tokens=64000,
                     temperature=0)
"""`claude-3-7-sonnet` model."""

claude4opus = LLM(model_name="claude-opus-4-20250514",
                   max_tokens=32000,
                   temperature=0)
"""`claude-4-Opus` model."""

claude41opus = LLM(model_name="claude-opus-4-1-20250805",
                   max_tokens=32000,
                   temperature=0)
"""`claude-4.1-Opus` model."""

vllm_llama3 = LLM(model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                  max_tokens=8192,
                  temperature=0.7,
                  model_type="local",
                  client="vllm")
"""vLLM Llama 3 model."""

ollama_llama3 = LLM(model_name="llama3",
                    max_tokens=8192,
                    temperature=0.7,
                    model_type="local",
                    client="ollama")
"""Ollama Llama 3 model."""

models : Dict[str, LLM] = {
                            "vllm-llama3": vllm_llama3,
                            "ollama-llama3": ollama_llama3,
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
