import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. Configuration ---
DIRECTORY = 'dataset/json_chunking'
TOKEN_THRESHOLD = 30  # The token count to define a "small sentence"
CHARS_PER_TOKEN = 4   # Your assumption: 4 characters per token
CHAR_THRESHOLD = TOKEN_THRESHOLD * CHARS_PER_TOKEN

# --- 2. Data Collection ---
all_sentence_lengths = []
total_sentences = 0
small_sentence_count = 0
files_processed = 0

print(f"Analyzing files in '{DIRECTORY}'...")
print(f"A 'small sentence' is defined as less than {TOKEN_THRESHOLD} tokens ({CHAR_THRESHOLD} characters).")

# Use os.listdir and wrap with tqdm for a progress bar
file_list = os.listdir(DIRECTORY)
for filename in tqdm(file_list, desc="Processing files"):
    if not filename.endswith('.json'):
        continue

    file_path = os.path.join(DIRECTORY, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sentence_list = data.get("merged_processed_sentences")
        if not sentence_list:
            continue
            
        # Calculate the length of each sentence in the current file
        lengths_in_file = [len(sentence) for sentence in sentence_list]
        
        # Add these lengths to our master list
        all_sentence_lengths.extend(lengths_in_file)
        
        # Count the small sentences in this file
        for length in lengths_in_file:
            if length < CHAR_THRESHOLD:
                small_sentence_count += 1
        
        total_sentences += len(sentence_list)
        files_processed += 1

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"\nSkipped {filename} due to error: {e}")

# --- 3. Statistical Summary ---
print("\n" + "="*50)
print("--- Analysis Complete ---")
if total_sentences > 0:
    percentage_small = (small_sentence_count / total_sentences) * 100
    print(f"Files Processed:          {files_processed}")
    print(f"Total Sentences Analyzed: {total_sentences:,}")
    print(f"Small Sentences (<{TOKEN_THRESHOLD} tokens): {small_sentence_count:,}")
    print(f"Percentage of Small Sentences: {percentage_small:.2f}%")
    print("="*50)

    # --- 4. Visualization ---
    print("\nGenerating sentence length distribution histogram...")
    plt.figure(figsize=(12, 6))
    plt.hist(all_sentence_lengths, bins=100, range=(0, 1000), color='skyblue', edgecolor='black')
    
    # Add a vertical line at the threshold
    plt.axvline(CHAR_THRESHOLD, color='r', linestyle='--', linewidth=2, label=f'<{TOKEN_THRESHOLD} Token Threshold ({CHAR_THRESHOLD} chars)')
    
    plt.title('Distribution of Sentence Lengths (in Characters)')
    plt.xlabel('Number of Characters per Sentence')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Save the plot to a file
    plt.savefig('sentence_length_distribution.png')
    print("Histogram saved to 'sentence_length_distribution.png'")
    
else:
    print("No valid sentences found to analyze.")