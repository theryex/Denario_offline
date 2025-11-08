from .denario import Denario, Research, Journal, LLM, KeyManager

def __getattr__(name):
    if name == "models":
        from .llm import models
        return models
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
