import requests
import ollama

def get_vllm_models(base_url: str) -> list[str]:
    """Get the list of available models from a vLLM server."""
    try:
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()
        models = response.json()
        return [model["id"] for model in models["data"]]
    except requests.exceptions.RequestException as e:
        print(f"Error getting vLLM models: {e}")
        return []

def get_ollama_models(host: str) -> list[str]:
    """Get the list of available models from an Ollama server."""
    try:
        client = ollama.Client(host=host)
        models = client.list()["models"]
        return [model["name"] for model in models]
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        return []
