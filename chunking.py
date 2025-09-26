import os
import json
import random
import time
import requests
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# Import the necessary library for threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. Configuration ---
INPUT_DIRECTORY = 'dataset/json_chunking'
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "qwen3-embedding:4b"
# The number of parallel requests to send to the Ollama API
# A good starting point is the number of cores you have.
NUM_THREADS = 4
SAMPLE_SIZE = 100
PERCENTILES_TO_TEST = [10, 15, 20, 25, 30, 40]

FILE_OUTPUT_DIRECTORY = "dataset"
FILE_NAME = "papers_list.txt"
FILES_LIST = []

# --- 2. Ollama Embedding Function (for a single text) ---
def get_ollama_embedding(text: str) -> list[float]:
    """Gets a single embedding from the Ollama API."""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.RequestException as e:
        # Return None on failure so we can handle it
        print(f"Warning: API call failed. Error: {e}")
        return None

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two numpy arrays."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0.0

def get_similarity_scores_ollama_parallel(sentences: list[str]) -> list[float]:
    """
    Calculates similarities by fetching embeddings in parallel using a thread pool.
    """
    if len(sentences) < 2:
        return []

    all_embeddings = [None] * len(sentences) # Pre-allocate list to maintain order
    
    # Use a ThreadPoolExecutor to make concurrent API calls
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Create a future for each sentence
        future_to_index = {executor.submit(get_ollama_embedding, sentence): i for i, sentence in enumerate(sentences)}
        
        # Use tqdm to show progress for embedding calls
        for future in tqdm(as_completed(future_to_index), total=len(sentences), desc="Embedding Sentences", leave=False):
            index = future_to_index[future]
            try:
                embedding = future.result()
                if embedding:
                    all_embeddings[index] = embedding
            except Exception as e:
                print(f"Error fetching embedding for sentence at index {index}: {e}")

    # Filter out any failed embedding calls (where the value is still None)
    valid_embeddings = [emb for emb in all_embeddings if emb is not None]
    
    if len(valid_embeddings) < 2:
        return []

    all_embeddings_np = np.array(valid_embeddings)
    
    similarities = []
    for i in range(len(all_embeddings_np) - 1):
        sim = cosine_similarity(all_embeddings_np[i], all_embeddings_np[i+1])
        similarities.append(sim)
        
    return similarities

# --- Main Execution Block ---
if __name__ == '__main__':
    # ... (The rest of your main script remains exactly the same) ...
    # It will now automatically call the new, parallelized function.
    
    start_time = time.time()
    
    try:
        requests.get(OLLAMA_HOST, timeout=5)
        print(f"Successfully connected to Ollama at {OLLAMA_HOST}")
    except requests.RequestException:
        print(f"Error: Could not connect to Ollama at {OLLAMA_HOST}. Please ensure it's running.")
        exit()

    all_files = [f for f in os.listdir(INPUT_DIRECTORY) if f.endswith('.json')]
    sample_files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    print(f"Selected a random sample of {len(sample_files)} files for analysis.")
    
    print("\nCalculating and pooling similarity scores using Ollama (in parallel)...")
    pooled_scores = []
    documents_sentences = {}

    for filename in tqdm(sample_files, desc="Processing sample files"):
        file_path = os.path.join(INPUT_DIRECTORY, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Currently working on file - {filename}")
        FILES_LIST.append(filename)
        sentences = data.get('merged_processed_sentences', [])
        print(f"Number of sentences in current file : {data.get('merged_processed_sentences_count', [])}")
        if sentences:
            documents_sentences[filename] = sentences
            # The only change needed here is calling the new parallel function
            scores = get_similarity_scores_ollama_parallel(sentences)
            if scores:
                pooled_scores.extend(scores)

    # --- The rest of the analysis (Steps 3 and 4) remains the same ---
    # ... (pasting the rest of the script for completeness) ...
    if not pooled_scores:
        print("Error: No similarity scores calculated. Check API connection or sentence data.")
    else:
        print("\n--- Global Distribution Analysis ---")
        candidate_thresholds = np.percentile(pooled_scores, PERCENTILES_TO_TEST)
        
        print("Candidate Thresholds based on Global Percentiles:")
        for p, t in zip(PERCENTILES_TO_TEST, candidate_thresholds):
            print(f"  - {p}th Percentile: {t:.4f}")

        plt.figure(figsize=(10, 6))
        plt.hist(pooled_scores, bins=100, color='teal', edgecolor='black')
        plt.title('Global Distribution of Sentence Similarity Scores (Ollama)')
        plt.xlabel('Cosine Similarity'); plt.ylabel('Frequency')
        plt.grid(True, alpha=0.5)
        for p, t in zip(PERCENTILES_TO_TEST, candidate_thresholds):
            plt.axvline(t, linestyle='--', alpha=0.8, label=f'{p}th Perc. ({t:.2f})')
        plt.legend()
        plt.savefig('ollama_similarity_distribution.png')
        print("\nSaved a histogram of the distribution to 'ollama_similarity_distribution.png'")

    print("\n--- Testing Candidate Thresholds ---")
    print("\n" + "="*60)
    print("Threshold | Avg. Chunks per Doc | Avg. Sentences per Chunk")
    print("-"*60)
    
    for percentile, threshold in zip(PERCENTILES_TO_TEST, candidate_thresholds):
        total_chunks = 0
        all_chunk_lengths = []
        
        for sentences in documents_sentences.values():
            if len(sentences) < 2:
                total_chunks +=1
                all_chunk_lengths.append(len(sentences))
                continue
            
            scores = get_similarity_scores_ollama_parallel(sentences)
            if not scores:
                total_chunks += 1
                all_chunk_lengths.append(len(sentences))
                continue

            num_chunks = 1
            current_chunk_len = 1
            for score in scores:
                if score < threshold:
                    all_chunk_lengths.append(current_chunk_len)
                    num_chunks += 1
                    current_chunk_len = 1
                else:
                    current_chunk_len += 1
            all_chunk_lengths.append(current_chunk_len)
            total_chunks += num_chunks

        avg_chunks = total_chunks / len(documents_sentences)
        avg_sents_chunk = sum(all_chunk_lengths) / len(all_chunk_lengths) if all_chunk_lengths else 0
        
        print(f"  {percentile}th ({threshold:.3f}) | {avg_chunks:20.2f} | {avg_sents_chunk:26.2f}")
    
    print("Writing filenames to text file ")
    text_file_path = os.path.join(FILE_OUTPUT_DIRECTORY, FILE_NAME)
    try:
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.writelines(f"{filename}\n" for filename in FILES_LIST)

        print(f"Successfully wrote {len(FILES_LIST)} filenames to '{text_file_path}'")

    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")
    print("="*60)
    end_time = time.time()
    print(f"\nAnalysis complete in {end_time - start_time:.2f} seconds.")