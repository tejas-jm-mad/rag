import os
import json
import re
import time
import yake
from keybert import KeyBERT
import spacy
import markdown
from bs4 import BeautifulSoup
import pytextrank

# --- Configuration ---
INPUT_DIRECTORY = "processed_markdown"
# Define a custom list of stop words to ignore during keyword extraction
CUSTOM_STOP_WORDS = {
    'abstract', 'introduction', 'conclusion', 'references', 'appendix',
    'arxiv', 'preprint', 'figure', 'fig', 'table', 'et', 'al', 'eq', 'equation',
    'paper', 'research', 'study', 'result', 'method', 'model'
}


# --- Advanced Preprocessing Function ---
def advanced_preprocess_text(markdown_text, nlp_processor):
    """
    Cleans text more effectively using lemmatization and custom stop words.
    - Strips all Markdown/HTML syntax
    - Removes irrelevant sections (e.g., references)
    - Uses spaCy for high-quality lemmatization
    - Filters out custom stop words
    """
    # 1. Convert Markdown to HTML and then to plain text
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()

    # 2. Remove irrelevant sections like the bibliography (case-insensitive)
    text = re.split(r'\nreferences|\nbibliography', text.lower())[0]

    # 3. Process with spaCy for high-quality tokenization and lemmatization
    doc = nlp_processor(text)
    
    lemmatized_tokens = []
    for token in doc:
        # 4. Keep tokens that are alphabetic, not standard stop words, and not in our custom list
        if token.is_alpha and not token.is_stop and token.lemma_ not in CUSTOM_STOP_WORDS:
            lemmatized_tokens.append(token.lemma_)
            
    return " ".join(lemmatized_tokens)


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. OPTIMIZATION: Load heavy models ONCE before the loop ---
    print("Loading models into memory... (This might take a moment)")
    # KeyBERT Model
    kw_model_keybert = KeyBERT(model="all-MiniLM-L6-v2")
    # Load one spaCy model for both preprocessing and TextRank
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    print("Models loaded successfully.")
    print("=" * 60)

    # --- 2. Initialize lists to store timings for averaging ---
    yake_times = []
    keybert_times = []
    textrank_times = []
    counter = 0

    # --- 3. Main processing loop ---
    for filename in os.listdir(INPUT_DIRECTORY):
        try:
            input_file_path = os.path.join(INPUT_DIRECTORY, filename)
            with open(input_file_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            # This print statement was moved outside the loop for cleaner output
            # print("Markdown content loaded successfully:") 

            if not markdown_content:
                print(f"--- Skipping: {filename} (Missing content) ---")
                continue

            # Use the new advanced preprocessing function
            cleaned_text = advanced_preprocess_text(markdown_content, nlp)

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
            doc = nlp(cleaned_text) # Re-use the nlp object
            duration = time.time() - start_time
            textrank_times.append(duration)

            print(f"TextRank Time: {duration:.4f}s | Keywords:")
            for phrase in doc._.phrases[:10]:
                print(f"  - {phrase.text} (Rank: {phrase.rank:.4f})")

            counter += 1
            if counter >= 10:
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