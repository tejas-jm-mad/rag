import os
import json
import re
import time
import yake
from keybert import KeyBERT
import spacy
import pytextrank

# --- Configuration ---
INPUT_DIRECTORY = "arxiv_preprocessed"


def preprocess_text(text):
    """
    Cleans and preprocesses the text for keyword extraction.
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. OPTIMIZATION: Load heavy models ONCE before the loop ---
    print("Loading models into memory... (This might take a moment)")
    # KeyBERT Model
    kw_model_keybert = KeyBERT(model="all-MiniLM-L6-v2")
    # spaCy with TextRank Model
    nlp_textrank = spacy.load("en_core_web_sm")
    nlp_textrank.add_pipe("textrank")
    print("Models loaded successfully.")
    print("=" * 60)

    # --- 2. Initialize lists to store timings for averaging ---
    yake_times = []
    keybert_times = []
    textrank_times = []
    counter = 0

    file_list = [f for f in os.listdir(INPUT_DIRECTORY) if f.endswith(".json")]

    # --- 3. Main processing loop ---
    for filename in file_list:
        try:
            input_file_path = os.path.join(INPUT_DIRECTORY, filename)
            with open(input_file_path, "r", encoding="utf-8") as file:
                original_info = json.load(file)

            paper_text = original_info.get("original_description", "")
            paper_abstract = original_info.get("abstract", "")

            if not paper_text or not paper_abstract:
                print(f"--- Skipping: {filename} (Missing content) ---")
                continue

            # Combine abstract and main text for keyword extraction
            combined_text = (
                "Abstract: " + paper_abstract + " Paper Content: " + paper_text
            )
            cleaned_text = preprocess_text(combined_text)

            print(f"\n--- Processing: {filename} ---")

            # --- Algorithm 1: YAKE! ---
            start_time = time.time()
            kw_extractor_yake = yake.KeywordExtractor(n=3, top=10)
            keywords_yake = kw_extractor_yake.extract_keywords(cleaned_text)
            duration = time.time() - start_time
            yake_times.append(duration)

            print(f"YAKE! Time: {duration:.4f}s | Keywords:")
            for kw, score in keywords_yake:
                print(f"  - {kw} (Score: {score:.4f})")

            # --- Algorithm 2: KeyBERT ---
            start_time = time.time()
            keywords_keybert = kw_model_keybert.extract_keywords(
                cleaned_text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=10,
            )
            duration = time.time() - start_time
            keybert_times.append(duration)

            print(f"KeyBERT Time: {duration:.4f}s | Keywords:")
            for kw, score in keywords_keybert:
                print(f"  - {kw} (Score: {score:.4f})")

            # --- Algorithm 3: TextRank (pytextrank) ---
            start_time = time.time()
            doc = nlp_textrank(cleaned_text)
            duration = time.time() - start_time
            textrank_times.append(duration)

            print(f"TextRank Time: {duration:.4f}s | Keywords:")
            for phrase in doc._.phrases[:10]:
                print(f"  - {phrase.text} (Rank: {phrase.rank:.4f})")

            counter += 1
            if counter >= 1:
                break

        except Exception as e:
            print(f"--- FAILED on {filename}: {e} ---")

    # --- 4. Calculate and print global averages ---
    print("\n" + "=" * 60)
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

    print("=" * 60)
