import os
import re
import time
import json
import yake
from keybert import KeyBERT
import spacy
import pytextrank
import markdown
from bs4 import BeautifulSoup

# --- Configuration ---
MARKDOWN_DIRECTORY = 'processed_markdown'
# Define a custom list of stop words to ignore during keyword extraction
CUSTOM_STOP_WORDS = {
    'abstract', 'introduction', 'conclusion', 'references', 'appendix',
    'arxiv', 'preprint', 'figure', 'fig', 'table', 'et', 'al', 'eq', 'equation',
    'paper', 'research', 'study', 'result', 'method', 'model'
}

# --- 1. Advanced Preprocessing Function (with Lemmatization) ---
def advanced_preprocess_text(markdown_text, nlp_processor):
    """
    Cleans text more effectively using lemmatization and custom stop words.
    Ideal for statistical/graph-based methods like YAKE! and TextRank.
    """
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    text = re.split(r'\nreferences|\nbibliography', text.lower())[0]
    doc = nlp_processor(text)
    lemmatized_tokens = []
    for token in doc:
        if token.is_alpha and not token.is_stop and token.lemma_ not in CUSTOM_STOP_WORDS:
            lemmatized_tokens.append(token.lemma_)
    return " ".join(lemmatized_tokens)

# --- 2. Simpler Preprocessing for KeyBERT ---
def simple_clean_for_keybert(markdown_text):
    """
    A lighter cleaning function for KeyBERT, which benefits from more natural text.
    """
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    text = re.split(r'\nreferences|\nbibliography', text)[0]
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- 1. Load heavy models ONCE ---
    print("Loading models into memory...")
    kw_model_keybert = KeyBERT(model='all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    print("Models loaded successfully.")
    print("="*60)

    # List of files to process
    markdown_files_to_process = [
        # "1706.03762v7_Attention Is All You Need.md",
        # "1701.06538v1_Outrageously Large Neural Networks_ The Sparsely-Gated   Mixture-of-Experts Layer.md",
        "2509.20279v1_A co-evolving agentic AI system for medical imaging analysis.md",
    ]

    # --- 2. Initialize lists to store timings for averaging ---
    yake_times, keybert_times, textrank_times = [], [], []

    # --- 3. Main processing loop ---
    for filename in markdown_files_to_process:
        try:
            input_file_path = os.path.join(MARKDOWN_DIRECTORY, filename)
            
            if not os.path.exists(input_file_path):
                print(f"--- Skipping: {filename} (File not found) ---")
                continue

            with open(input_file_path, 'r', encoding='utf-8') as file:
                markdown_content = file.read()
            
            if not markdown_content.strip():
                print(f"--- Skipping: {filename} (File is empty) ---")
                continue
            
            # --- Prepare different versions of the text ---
            advanced_cleaned_text = advanced_preprocess_text(markdown_content, nlp)
            simple_cleaned_text = simple_clean_for_keybert(markdown_content)

            print(f"\n--- Processing: {filename} ---")

            # --- Algorithm 1: YAKE! ---
            start_time = time.time()
            kw_extractor_yake = yake.KeywordExtractor(n=3, top=10, dedupLim=0.9)
            keywords_yake = kw_extractor_yake.extract_keywords(advanced_cleaned_text)
            duration = time.time() - start_time
            yake_times.append(duration)
            print(f"YAKE! Time: {duration:.4f}s | Keywords (Improved):")
            for kw, score in keywords_yake:
                print(f"  - {kw}")
            
            # --- Algorithm 2: KeyBERT ---
            start_time = time.time()
            keywords_keybert = kw_model_keybert.extract_keywords(simple_cleaned_text,
                                                                 keyphrase_ngram_range=(1, 3),
                                                                 stop_words='english',
                                                                 use_mmr=True,
                                                                 diversity=0.7,
                                                                 top_n=10)
            duration = time.time() - start_time
            keybert_times.append(duration)
            print(f"\nKeyBERT Time: {duration:.4f}s | Keywords (with MMR for Diversity):")
            for kw, score in keywords_keybert:
                print(f"  - {kw} (Score: {score:.4f})")
            
            # --- Algorithm 3: TextRank ---
            start_time = time.time()
            doc = nlp(advanced_cleaned_text)
            duration = time.time() - start_time
            textrank_times.append(duration)
            print(f"\nTextRank Time: {duration:.4f}s | Keywords (Improved):")
            for phrase in doc._.phrases[:10]:
                print(f"  - {phrase.text}")

        except Exception as e:
            print(f"--- FAILED on {filename}: {e} ---")

    # --- 4. Calculate and print global averages ---
    print("\n" + "="*60)
    print("--- Global Performance Summary ---")
    
    if yake_times:
        avg_yake = sum(yake_times) / len(yake_times)
        print(f"Average YAKE! Time per file: {avg_yake:.4f} seconds")
    
    if keybert_times:
        avg_keybert = sum(keybert_times) / len(keybert_times)
        print(f"Average KeyBERT Time per file: {avg_keybert:.4f} seconds")

    if textrank_times:
        avg_textrank = sum(textrank_times) / len(textrank_times)
        print(f"Average TextRank Time per file: {avg_textrank:.4f} seconds")
    
    print("="*60)