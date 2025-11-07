from .llm import LLM, models
import requests

def get_vllm_models(base_url: str):
    """Get available vLLM models from the API."""
    try:
        response = requests.get(f"{base_url}/models")
        response.raise_for_status()
        return [model['id'] for model in response.json()['data']]
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not connect to vLLM API at {base_url}. vLLM models will not be available. Error: {e}")
        return []

def get_ollama_models(host: str):
    """Get available Ollama models from the API."""
    try:
        response = requests.get(f"{host}/api/tags")
        response.raise_for_status()
        return [model['name'] for model in response.json()['models']]
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not connect to Ollama API at {host}. Ollama models will not be available. Error: {e}")
        return []

def update_models_with_local_llms(vllm_base_url: str | None = None, ollama_host: str | None = None):
    """Update the models dictionary with available local LLMs."""
    if vllm_base_url:
        vllm_models = get_vllm_models(vllm_base_url)
        for model_name in vllm_models:
            models[f"vllm-{model_name}"] = LLM(name=model_name,
                                               max_output_tokens=8192,
                                               temperature=0.7,
                                               model_type="local",
                                               client="vllm")
    if ollama_host:
        ollama_models = get_ollama_models(ollama_host)
        for model_name in ollama_models:
            models[f"ollama-{model_name}"] = LLM(name=model_name,
                                                 max_output_tokens=8192,
                                                 temperature=0.7,
                                                 model_type="local",
                                                 client="ollama")
