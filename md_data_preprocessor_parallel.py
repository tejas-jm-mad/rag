import os
import re
import json
import spacy
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import markdown as md
from bs4 import BeautifulSoup
from ftfy import fix_text
import html as htmllib
import unicodedata
import codecs

# --- Configuration ---
METADATA_DIRECTORY = 'dataset/api_metadata'
MARKDOWN_DIRECTORY = 'dataset/markdown'
OUTPUT_DIRECTORY = 'dataset/json_chunking'
MIN_SENTENCE_MERGE_LENGTH = 120 

# --- Worker Process Initialization ---
nlp = None

def init_worker():
    """Initializes a spaCy model for each worker process."""
    global nlp
    print("Initializing a worker process...")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")

# --- Sentence Merging Function ---
def merge_small_sentences(sentence_list, min_length=80):
    """
    Merges small, adjacent sentences into larger ones using a forward-merge approach.
    """
    if not sentence_list:
        return []

    merged_sentences = []
    current_chunk = ""

    for sentence in sentence_list:
        current_chunk += " " + sentence.strip()
        if len(current_chunk) >= min_length:
            merged_sentences.append(current_chunk.strip())
            current_chunk = ""
            
    # Add the last remaining chunk if it's not empty
    if current_chunk.strip():
        merged_sentences.append(current_chunk.strip())
        
    return merged_sentences

# --- Preprocessing function for sentence splitting ---
def preprocess_markdown_for_sentencizing(
    markdown_text: str,
    *,
    unicode_mode: str = "keep",     # "keep", "strip", "ascii", "german"
    keep_emoji: bool = True
) -> str:
    """
    Cleans Markdown to plain text optimized for sentence splitting, with robust Unicode handling.

    - Removes code blocks/inline code
    - Keeps human-visible text for links/images; drops raw URLs
    - Markdown -> HTML -> text, preserving block boundaries as newlines
    - Uses ftfy (if available) to fix mojibake
    - Canonicalizes Unicode (NFKC), maps smart punctuation, strips zero-width/bidi/VS chars
    - **Composes combining marks (NFC) and removes stray combining marks (e.g., U+0301 alone)**
    - Optional emoji removal
    - Optional accent stripping or German transliteration (ä->ae, ß->ss)
    - Collapses whitespace while preserving paragraph breaks
    """

    if not markdown_text:
        return ""

    # --- Compile once ---
    CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
    INLINE_CODE_RE = re.compile(r"`[^`]+`")
    IMAGE_RE      = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
    LINK_RE       = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
    BARE_URL_RE   = re.compile(r"\b(?:https?|ftp)://\S+\b", re.IGNORECASE)
    WS_LINE_RE    = re.compile(r"[ \t]+")
    WS_ANY_RE     = re.compile(r"\s+")
    ESCAPED_UNICODE_RE = re.compile(r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}")

    # Combining mark ranges (Mn/Mc/Me) to help detect “stray” accents
    COMBINING_RANGES = (
        (0x0300, 0x036F),  # Combining Diacritical Marks
        (0x1AB0, 0x1AFF),  # Combining Diacritical Marks Extended
        (0x1DC0, 0x1DFF),  # Combining Diacritical Marks Supplement
        (0x20D0, 0x20FF),  # Combining Diacritical Marks for Symbols
        (0xFE20, 0xFE2F),  # Combining Half Marks
    )
    def _is_combining(ch: str) -> bool:
        cp = ord(ch)
        for a, b in COMBINING_RANGES:
            if a <= cp <= b:
                return True
        return unicodedata.category(ch).startswith("M")

    PUNCT_MAP = {
        0x2018:"'", 0x2019:"'", 0x201A:"'", 0x201B:"'",
        0x201C:'"', 0x201D:'"', 0x201E:'"', 0x201F:'"',
        0x00AB:'"',  0x00BB:'"',
        0x2013:"-",  0x2014:"-", 0x2212:"-",
        0x2026:"...",
    }
    GERMAN_MAP = str.maketrans({
        "ä":"ae", "ö":"oe", "ü":"ue",
        "Ä":"Ae", "Ö":"Oe", "Ü":"Ue",
        "ß":"ss",
    })
    ZW_CHARS = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E}
    BIDI_MARKS = {0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}
    VARIATION_SELECTORS = set(range(0xFE00, 0xFE10)) | {0xFE0F}

    # --- Helpers ---
    def maybe_decode_escapes(s: str) -> str:
        if ESCAPED_UNICODE_RE.search(s):
            try:
                return codecs.decode(s, "unicode_escape")
            except Exception:
                return s
        return s

    def strip_controls(s: str) -> str:
        return "".join(
            ch for ch in s
            if not ((unicodedata.category(ch) in ("Cc", "Cf")) and ch not in ("\n", "\t"))
        )

    def remove_emoji(s: str) -> str:
        def is_emoji(ch):
            cp = ord(ch)
            return (
                0x1F000 <= cp <= 0x1FAFF or
                0x2600  <= cp <= 0x27BF  or
                cp == 0xFE0F
            )
        return "".join(ch for ch in s if not is_emoji(ch))

    def canonicalize_unicode(s: str) -> str:
        # Try ftfy to fix mojibake
        try:
            s = fix_text(s)
        except Exception:
            pass

        # Decode entities + normalize
        s = htmllib.unescape(s)
        s = unicodedata.normalize("NFKC", s)

        # ASCII-friendly punctuation
        s = s.translate(PUNCT_MAP)

        # Remove zero-width/bidi/variation selectors
        s = "".join(
            ch for ch in s
            if (ord(ch) not in ZW_CHARS) and (ord(ch) not in BIDI_MARKS) and (ord(ch) not in VARIATION_SELECTORS)
        )

        if not keep_emoji:
            s = remove_emoji(s)

        # Standardize whitespace
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = WS_ANY_RE.sub(lambda m: "\n" if "\n" in m.group(0) else " ", s)
        return s

    def compose_and_drop_stray_combining(s: str) -> str:
        """
        Compose to NFC so base+combining -> precomposed if possible,
        then remove combining marks that don't follow a base letter/number mark.
        """
        s = unicodedata.normalize("NFC", s)

        out = []
        prev_is_base = False
        for ch in s:
            cat = unicodedata.category(ch)
            if _is_combining(ch):
                # keep only if attached to a preceding base character
                if prev_is_base:
                    out.append(ch)
                # else drop (this removes stray U+0301 etc.)
                continue
            else:
                out.append(ch)
                # treat letters and numbers as valid "bases"
                if cat.startswith("L") or cat.startswith("N"):
                    prev_is_base = True
                else:
                    # punctuation, symbols, separators break the base
                    prev_is_base = False
        return "".join(out)

    def unicode_postprocess(s: str) -> str:
        # Decode \uXXXX escapes if present
        s = maybe_decode_escapes(s)
        # Canonical cleanup
        s = canonicalize_unicode(s)
        # Compose accents and drop stray combining marks
        s = compose_and_drop_stray_combining(s)

        # Diacritics policy
        if unicode_mode == "german":
            s = s.translate(GERMAN_MAP)
            s = unicodedata.normalize("NFKC", s)
        elif unicode_mode in ("strip", "ascii"):
            s = unicodedata.normalize("NFKD", s)
            s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
            if unicode_mode == "ascii":
                s = s.encode("ascii", "ignore").decode("ascii")
        # "keep": leave as-is (already composed & de-strayed)
        return s

    # --- 1) Remove code that confuses sentence boundaries ---
    s = markdown_text
    s = CODE_FENCE_RE.sub(" ", s)
    s = INLINE_CODE_RE.sub(" ", s)

    # --- 2) Preserve human text for images/links; drop raw URLs ---
    s = IMAGE_RE.sub(r"\1", s)     # keep alt text
    s = LINK_RE.sub(r"\1", s)      # keep link text
    s = BARE_URL_RE.sub(" ", s)    # remove bare URLs

    # --- 3) Markdown -> HTML -> text with explicit separators ---
    html = md.markdown(s, extensions=["extra", "sane_lists", "tables", "nl2br"])
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # --- 4) Unicode cleanup / normalization ---
    text = strip_controls(text)
    text = unicode_postprocess(text)

    # --- 5) Collapse multiple blank lines; trim intra-line spaces ---
    out_lines = []
    for raw in text.splitlines():
        line = WS_LINE_RE.sub(" ", raw).strip()
        if line:
            out_lines.append(line)
        else:
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
    text = "\n".join(out_lines).strip()

    return text

# --- Worker Function ---
def process_metadata_file(metadata_filename):
    """
    Processes one metadata file: finds the matching markdown, processes it,
    and saves the combined result.
    """
    if not metadata_filename.endswith('.json'):
        return None

    try:
        # 1. Find the corresponding Markdown file
        markdown_path = f"dataset/markdown/{metadata_filename[:-5]}.md"
        if not markdown_path:
            return (metadata_filename, "Skipped", f"No matching .md file found for ID {metadata_filename}")

        # 3. Read both the metadata and markdown files
        metadata_path = os.path.join(METADATA_DIRECTORY, metadata_filename)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_info = json.load(f)
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()

        # 4. Preprocess the Markdown content
        cleaned_text = preprocess_markdown_for_sentencizing(markdown_content, unicode_mode="strip", keep_emoji=False)

        # 5. Perform sentence splitting
        doc = nlp(cleaned_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        merged_sentences = merge_small_sentences(sentences, min_length=MIN_SENTENCE_MERGE_LENGTH)
        
        # 6. Add all data to the original metadata
        metadata_info['processed_markdown'] = cleaned_text
        metadata_info['processed_sentences'] = sentences
        metadata_info['processed_sentences_count'] = len(sentences)
        metadata_info['merged_processed_sentences'] = merged_sentences
        metadata_info['merged_processed_sentences_count'] = len(merged_sentences)
        
        
        # 7. Write the result to the output directory
        output_path = os.path.join(OUTPUT_DIRECTORY, metadata_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_info, f, indent=4)
            
        return (metadata_filename, "Success", len(sentences))

    except Exception as e:
        return (metadata_filename, "Failed", str(e))
    

# --- Main Execution Block ---
if __name__ == '__main__':
    start_time = time.perf_counter()

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    if not os.path.isdir(METADATA_DIRECTORY):
        print(f"Error: Metadata directory '{METADATA_DIRECTORY}' not found.")
    else:
        file_list = sorted(os.listdir(METADATA_DIRECTORY))
        num_processes = cpu_count()
        print(f"Starting parallel processing on {num_processes} cores for {len(file_list)} files...")

        success_count = 0
        failed_count = 0
        
        with Pool(processes=num_processes, initializer=init_worker) as pool:
            for result in tqdm(pool.imap_unordered(process_metadata_file, file_list), total=len(file_list)):
                if result is None:
                    continue
                
                filename, status, message = result
                if status == "Success":
                    success_count += 1
                else:
                    failed_count += 1
                    print(f"Error/Skip on {filename}: {message}")

        print(f"\nFinished processing.")
        print(f"Total files successfully processed: {success_count}")
        print(f"Total files failed or skipped: {failed_count}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")


# Small Spacy Model
# Finished processing.
# Total files successfully processed: 253
# Total files failed or skipped: 0
# Execution time: 41.113254 seconds

# Large Spacy Model
# Finished processing.
# Total files successfully processed: 253
# Total files failed or skipped: 0
# Execution time: 49.745186 seconds