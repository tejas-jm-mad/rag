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

# --- Worker Process Initialization ---
nlp = None

def init_worker():
    """Initializes a spaCy model for each worker process."""
    global nlp
    print("Initializing a worker process...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")

# --- Preprocessing function for sentence splitting ---
def preprocess_markdown_for_sentencizing(
    markdown_text: str,
    *,
    unicode_mode: str = "keep",     # "keep", "strip", "ascii", "german"
    keep_emoji: bool = True
) -> str:
    """
    Cleans Markdown to plain text optimized for sentence splitting, with robust Unicode handling.

    Features:
      - Removes code blocks/inline code
      - Keeps human-visible text for links/images; drops raw URLs
      - Markdown -> HTML -> text, preserving block boundaries as newlines
      - Uses ftfy (if available) to fix mojibake and broken encodings
      - Canonicalizes Unicode (NFKC), maps smart punctuation, strips zero-width/bidi/variation selectors
      - Optional emoji removal
      - Optional accent stripping or German transliteration (ä->ae, ß->ss)
      - Collapses whitespace while preserving paragraph breaks

    Args:
      markdown_text: Source Markdown.
      unicode_mode:
         - "keep": keep accents (ä stays ä).
         - "strip": remove diacritics (ä -> a) via NFKD + drop combining marks.
         - "ascii": best-effort ASCII (ä -> a) and drop non-ASCII.
         - "german": transliterate German umlauts/ß (ä->ae, ö->oe, ü->ue, ß->ss).
      keep_emoji: keep emoji if True; otherwise remove common emoji chars.

    Returns:
      Clean plain text suitable for sentence splitting.
    """

    # --- Early exit ---
    if not markdown_text:
        return ""

    # --- Compile once (function scope so it's still "one function" public API) ---
    CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
    INLINE_CODE_RE = re.compile(r"`[^`]+`")
    IMAGE_RE      = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
    LINK_RE       = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
    BARE_URL_RE   = re.compile(r"\b(?:https?|ftp)://\S+\b", re.IGNORECASE)
    WS_LINE_RE    = re.compile(r"[ \t]+")
    WS_ANY_RE     = re.compile(r"\s+")
    ESCAPED_UNICODE_RE = re.compile(r"\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}")

    PUNCT_MAP = {
        0x2018:"'", 0x2019:"'", 0x201A:"'", 0x201B:"'",
        0x201C:'"', 0x201D:'"', 0x201E:'"', 0x201F:'"',
        0x00AB:'"',  0x00BB:'"',
        0x2013:"-",  0x2014:"-", 0x2212:"-",
        0x2026:"...",  # ellipsis
    }
    GERMAN_MAP = str.maketrans({
        "ä":"ae", "ö":"oe", "ü":"ue",
        "Ä":"Ae", "Ö":"Oe", "Ü":"Ue",
        "ß":"ss",
    })
    ZW_CHARS = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E}
    BIDI_MARKS = {0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}
    VARIATION_SELECTORS = set(range(0xFE00, 0xFE10)) | {0xFE0F}

    # --- Helper closures (kept inside to honor "single function" request) ---
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
                0x1F000 <= cp <= 0x1FAFF or  # emoji blocks
                0x2600  <= cp <= 0x27BF  or  # dingbats/misc
                cp == 0xFE0F                 # VS16 (emoji presentation)
            )
        return "".join(ch for ch in s if not is_emoji(ch))

    def canonicalize_unicode(s: str) -> str:
        # 0) Fix broken encodings if ftfy is available
        try:
            s = fix_text(s)
        except Exception:
            pass

        # 1) Decode entities and normalize
        s = htmllib.unescape(s)
        s = unicodedata.normalize("NFKC", s)

        # 2) Replace curly punctuation with ASCII
        s = s.translate(PUNCT_MAP)

        # 3) Drop zero-width, bidi marks, and variation selectors
        s = "".join(
            ch for ch in s
            if (ord(ch) not in ZW_CHARS) and (ord(ch) not in BIDI_MARKS) and (ord(ch) not in VARIATION_SELECTORS)
        )

        # 4) Optionally drop emoji
        if not keep_emoji:
            s = remove_emoji(s)

        # 5) Whitespace normalize (standardize \r to \n, then collapse runs)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = WS_ANY_RE.sub(lambda m: "\n" if "\n" in m.group(0) else " ", s)
        return s

    def unicode_postprocess(s: str) -> str:
        # Handle literal escape sequences like "\u00e4" -> "ä"
        s = maybe_decode_escapes(s)

        # Canonical cleanup
        s = canonicalize_unicode(s)

        # Transliteration / diacritics handling
        if unicode_mode == "german":
            s = s.translate(GERMAN_MAP)
            s = unicodedata.normalize("NFKC", s)
        elif unicode_mode in ("strip", "ascii"):
            s = unicodedata.normalize("NFKD", s)
            s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
            if unicode_mode == "ascii":
                s = s.encode("ascii", "ignore").decode("ascii")
        # else: "keep" -> do nothing further
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

    # --- 5) Collapse multiple blank lines to at most one; trim intra-line spaces ---
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
        cleaned_text = preprocess_markdown_for_sentencizing(markdown_content)

        # 5. Perform sentence splitting
        doc = nlp(cleaned_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        num_sentences = len(sentences)

        # 6. Add the new data to the original metadata
        metadata_info['processed_sentences'] = sentences
        metadata_info['processed_sentences_count'] = num_sentences
        
        # 7. Write the result to the output directory
        output_path = os.path.join(OUTPUT_DIRECTORY, metadata_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_info, f, indent=4)
            
        return (metadata_filename, "Success", num_sentences)

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


# Finished processing.
# Total files successfully processed: 253
# Total files failed or skipped: 0
# Execution time: 41.113254 seconds