import requests
import json

# The single Ollama server address remains the same
OLLAMA_HOST = "http://localhost:11434"

# --- Function to use the NEW Embedding Model ---
def get_embedding(text: str) -> list[float]:
    """Gets an embedding from the 'mxbai-embed-large' model."""
    response = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={
            "model": "qwen3-embedding:4b",  # <-- Use the new embedding model name
            "prompt": text
        }
    )
    response.raise_for_status()
    return response.json()["embedding"]

# --- Function to use the NEW Filter Extraction SLM ---
def extract_filters_from_query(query: str) -> dict:
    """Extracts filters using the 'phi3' model."""
    # The prompt engineering strategy remains the same
    prompt = f"You are an expert at extracting JSON filters... (rest of your prompt) \nUser Query: \"{query}\"\nJSON:"
    
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": "phi3:latest",  # <-- Use the new SLM name
            "prompt": prompt,
            "format": "json", # This helps ensure the output is valid JSON
            "stream": False
        }
    )
    response.raise_for_status()
    try:
        return json.loads(response.json()['response'])
    except json.JSONDecodeError:
        return {} # Return empty on failure