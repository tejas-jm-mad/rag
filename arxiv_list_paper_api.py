import requests
import xml.etree.ElementTree as ET
import re
import json
import time
from pathlib import Path
from docling.document_converter import DocumentConverter
import warnings
from pypdf import PdfReader

# Suppress the specific UserWarning from PyTorch's DataLoader about 'pin_memory'
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# ==============================================================================
# == CONFIGURATION -
# ==============================================================================
# Add the unique arXiv IDs of the papers you want to process here.
PAPER_IDS = [
    "2408.09869",  # Docling
    "1706.03762",  # Attention Is All You Need
    "1701.06538",  # MoE
]

# --- Directory Paths ---
PDF_SAVE_DIR = "downloaded_pdfs"
MD_SAVE_DIR = "processed_markdown"
METADATA_SAVE_DIR = "paper_metadata"

# ==============================================================================
# == SCRIPT LOGIC
# ==============================================================================


def _parse_entry_to_dict(entry, namespace):
    """Helper to parse an XML entry into a metadata dictionary."""

    def get_text(element, tag):
        found = element.find(tag, namespace)
        return (
            found.text.strip().replace("\n", " ")
            if found is not None and found.text
            else None
        )

    paper_id_raw = get_text(entry, "atom:id")
    paper_id = paper_id_raw.split("/abs/")[-1]

    return {
        "id": paper_id,
        "versioned_id": paper_id_raw.split("/")[-1],
        "title": get_text(entry, "atom:title"),
        "abstract": get_text(entry, "atom:summary"),
        "authors": [
            author.find("atom:name", namespace).text
            for author in entry.findall("atom:author", namespace)
        ],
        "published_date": get_text(entry, "atom:published"),
        "updated_date": get_text(entry, "atom:updated"),
        "primary_category": entry.find("arxiv:primary_category", namespace).get("term"),
        "categories": [
            cat.get("term") for cat in entry.findall("atom:category", namespace)
        ],
        "pdf_link": entry.find('atom:link[@title="pdf"]', namespace).get("href"),
        "abstract_link": entry.find('atom:link[@rel="alternate"]', namespace).get(
            "href"
        ),
        "doi": get_text(entry, "arxiv:doi"),
        "journal_ref": get_text(entry, "arxiv:journal_ref"),
    }


def fetch_and_save_metadata(id_list):
    """Fetches metadata from arXiv for a list of IDs and saves each as a JSON."""
    print(f"Fetching metadata for {len(id_list)} paper(s)...")
    id_string = ",".join(id_list)
    base_url = f"http://export.arxiv.org/api/query?id_list={id_string}"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed: {e}")
        return []

    root = ET.fromstring(response.content)
    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    entries = root.findall("atom:entry", namespace)

    papers_for_processing = []
    for entry in entries:
        metadata = _parse_entry_to_dict(entry, namespace)
        paper_id_sanitized = metadata["id"].replace("/", "_")

        # Save metadata to JSON
        filename = f"arxiv{paper_id_sanitized}_metadata.json"
        save_path = Path(METADATA_SAVE_DIR) / filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"  - Metadata for {metadata['id']} saved to JSON.")

        papers_for_processing.append(metadata)

    return papers_for_processing


def sanitize_filename(name):
    """Removes characters that are invalid in filenames."""
    return re.sub(r'[\/:*?"<>|]', "_", name)


def download_pdf(paper_info):
    """Downloads a PDF from a URL."""
    filename = f"{paper_info['id'].replace('/', '_')}_{sanitize_filename(paper_info['title'])}.pdf"
    pdf_path = Path(PDF_SAVE_DIR) / filename

    print(f"  - Downloading PDF for {paper_info['id']}...")
    try:
        response = requests.get(paper_info["pdf_link"], stream=True)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  - PDF saved to {pdf_path}")
        return pdf_path
    except requests.exceptions.RequestException as e:
        print(f"  - Error downloading PDF for {paper_info['id']}: {e}")
        return None


def process_with_docling(pdf_path):
    """Processes a PDF file with Docling and saves it as Markdown."""
    print(f"  - Processing {pdf_path.name} with Docling...")
    md_path = Path(MD_SAVE_DIR) / (pdf_path.stem + ".md")
    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown_content = result.document.export_to_markdown()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"  - Markdown saved to {md_path}")
        return md_path
    except Exception as e:
        print(f"  - Error processing with Docling for {pdf_path.name}: {e}")
        return None


def main():
    """Main function to orchestrate the workflow."""
    print("--- Starting arXiv Paper Processing Script ---")

    # Create all necessary directories
    Path(PDF_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(MD_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(METADATA_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Fetch metadata for all papers and save JSON files
    papers_to_process = fetch_and_save_metadata(PAPER_IDS)

    if not papers_to_process:
        print("--- No metadata found for the given IDs. Exiting. ---")
        return

    print("\n--- Starting Download and Processing Phase ---")
    # 2. Sequentially download and process each paper
    for i, paper in enumerate(papers_to_process, 1):
        print(
            f"\nProcessing paper {i}/{len(papers_to_process)}: {paper['id']} ({paper['title']})"
        )

        # Download the PDF
        pdf_path = download_pdf(paper)

        # If download is successful, process with Docling
        if pdf_path:
            process_with_docling(pdf_path)

    print("\n--- Script finished ---")


if __name__ == "__main__":
    main()
