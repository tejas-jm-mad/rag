import requests
import xml.etree.ElementTree as ET
import re
import time
import json  # <-- ADD THIS IMPORT
from pathlib import Path
from datetime import datetime, timedelta
from docling.document_converter import DocumentConverter
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from pypdf import PdfReader

warnings.filterwarnings("ignore", message=".*pin_memory.*")

# ==============================================================================
# == CONFIGURATION - SET YOUR PARAMETERS HERE
# ==============================================================================
MAX_WORKERS = 5
SEARCH_CATEGORY = "cs.AI"
MAX_RESULTS = 2
DAYS_TO_SEARCH = 300

# --- Directory Paths ---
PDF_SAVE_DIR = "downloaded_pdfs_dataset"
MD_SAVE_DIR = "processed_markdown_dataset"
METADATA_SAVE_DIR = "paper_metadata_dataset"

# ==============================================================================
# == SCRIPT LOGIC ==
# ==============================================================================

end_date_obj = datetime.now()
start_date_obj = end_date_obj - timedelta(days=DAYS_TO_SEARCH - 1)
START_DATE = start_date_obj.strftime("%Y-%m-%d")
END_DATE = end_date_obj.strftime("%Y-%m-%d")


def get_pdf_page_count(pdf_path):
    """Counts the number of pages in a PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except Exception as e:
        print(f"[Warning] Could not read page count for {Path(pdf_path).name}: {e}")
        return 0


def _parse_entry_to_dict(entry, namespace):
    """Helper function to parse a single XML entry into a metadata dictionary."""

    def get_text(element, tag):
        found = element.find(tag, namespace)
        return (
            found.text.strip().replace("\n", " ")
            if found is not None and found.text
            else None
        )

    paper_id_raw = get_text(entry, "atom:id")
    paper_id = paper_id_raw.split("/abs/")[-1]

    details = {
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
    return details


def search_arxiv(category, start_date, end_date, max_results):
    """Searches arXiv, saves metadata to JSON, and returns a list for processing."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d000000")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d235959")
    query = f"search_query=cat:{category}+AND+submittedDate:[{start_dt}+TO+{end_dt}]&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    base_url = "http://export.arxiv.org/api/query?"
    print(
        f"Querying arXiv API for category '{category}' from {start_date} to {end_date}."
    )

    try:
        response = requests.get(base_url + query)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed: {e}")
        return []

    root = ET.fromstring(response.content)
    # Add the 'arxiv' namespace to parse all metadata tags correctly
    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    entries = root.findall("atom:entry", namespace)

    if not entries:
        print("Search complete. No matching papers found.")
        return []

    print(f"Search complete. Found {len(entries)} papers. Saving metadata...")
    papers_for_processing = []
    for entry in entries:
        metadata = _parse_entry_to_dict(entry, namespace)

        # Save the full metadata to a JSON file
        paper_id_sanitized = metadata["id"].replace("/", "_")  # Sanitize for old IDs
        filename = f"arxiv{paper_id_sanitized}_metadata.json"
        save_path = Path(METADATA_SAVE_DIR) / filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        # Add the essential info to the list for the download/process workers
        papers_for_processing.append(
            {
                "id": metadata["id"],
                "title": metadata["title"],
                "pdf_url": metadata["pdf_link"],
            }
        )

    print(
        f"Metadata for {len(entries)} papers saved to '{METADATA_SAVE_DIR}' directory."
    )
    return papers_for_processing


def sanitize_filename(name):
    """Removes characters that are invalid in filenames."""
    return re.sub(r'[\/:*?"<>|]', "_", name)


def download_pdf(paper_info, save_dir):
    """Downloads a PDF from a URL and saves it to a directory."""
    filename = f"{paper_info['id'].replace('/', '_')}_{sanitize_filename(paper_info['title'])}.pdf"
    pdf_path = Path(save_dir) / filename
    try:
        response = requests.get(paper_info["pdf_url"], stream=True)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(pdf_path)
    except requests.exceptions.RequestException:
        return None


def process_with_docling(pdf_path, save_dir):
    """Processes a PDF file with Docling and saves it as Markdown."""
    pdf_path_obj = Path(pdf_path)
    md_path = Path(save_dir) / (pdf_path_obj.stem + ".md")
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown_content = result.document.export_to_markdown()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return str(md_path)
    except Exception:
        return None


def download_and_process_paper(paper_info):
    """A single task that downloads, processes, and times a paper."""
    start_time = time.monotonic()
    paper_id = paper_info["id"]
    print(f"[Worker] Task started for paper {paper_id}: Downloading PDF.")

    pdf_path_str = download_pdf(paper_info, PDF_SAVE_DIR)

    if pdf_path_str:
        page_count = get_pdf_page_count(pdf_path_str)
        print(
            f"[Worker] Download of {paper_id} ({page_count} pages) complete. Starting analysis."
        )
        md_path_str = process_with_docling(pdf_path_str, MD_SAVE_DIR)
        duration = time.monotonic() - start_time
        if md_path_str:
            return {
                "status": "success",
                "paper_id": paper_id,
                "duration": duration,
                "pages": page_count,
            }
        else:
            return {
                "status": "processing_failed",
                "paper_id": paper_id,
                "duration": duration,
                "pages": page_count,
            }
    else:
        duration = time.monotonic() - start_time
        return {
            "status": "download_failed",
            "paper_id": paper_id,
            "duration": duration,
            "pages": 0,
        }


def main():
    """Main function to orchestrate the parallel workflow."""
    workflow_start_time = time.monotonic()
    print("--- arXiv Paper Processing Workflow Initialized ---")

    # Create all necessary directories
    Path(PDF_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(MD_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(METADATA_SAVE_DIR).mkdir(
        parents=True, exist_ok=True
    )  # <-- CREATE METADATA DIR

    papers_to_process = search_arxiv(SEARCH_CATEGORY, START_DATE, END_DATE, MAX_RESULTS)

    if not papers_to_process:
        print("--- Workflow terminated. ---")
        return

    success_count = 0
    failure_count = 0
    total_pages_processed = 0
    total_time_spent_processing = 0.0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        print(
            f"\nSubmitting {len(papers_to_process)} download/process tasks to a pool of {MAX_WORKERS} parallel processes..."
        )

        future_tasks = {
            executor.submit(download_and_process_paper, paper): paper
            for paper in papers_to_process
        }

        for future in as_completed(future_tasks):
            try:
                result = future.result()
                duration = result["duration"]
                paper_id = result["paper_id"]

                if result["status"] == "success":
                    success_count += 1
                    pages = result["pages"]
                    total_pages_processed += pages
                    total_time_spent_processing += duration
                    print(
                        f"Success: Paper {paper_id} ({pages} pages) processed in {duration:.2f} seconds."
                    )
                else:
                    failure_count += 1
                    status_message = result["status"].replace("_", " ").capitalize()
                    print(
                        f"Failure: {status_message} for paper {paper_id}. Took {duration:.2f} seconds."
                    )
            except Exception as exc:
                failure_count += 1
                paper = future_tasks[future]
                print(
                    f"Error: An exception occurred while processing paper {paper['id']}: {exc}"
                )

    workflow_duration = time.monotonic() - workflow_start_time

    print("\n--------------------------------------------------")
    print("              Workflow Summary Report             ")
    print("--------------------------------------------------")
    print(f"Total execution time: {workflow_duration:.2f} seconds.")
    print(f"Successfully processed: {success_count} papers.")
    print(f"Failed to process:      {failure_count} papers.")
    print(f"Total pages analyzed:   {total_pages_processed}.")

    if total_pages_processed > 0:
        average_time_per_page = total_time_spent_processing / total_pages_processed
        print(f"Average time per page:  {average_time_per_page:.2f} seconds.")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()

# --- Workflow finished in 289.05 seconds! ---
# 📊 Global average time per page: 6.85 seconds

# --- Workflow finished in 263.92 seconds! ---
# 📊 Global average time per page: 5.60 seconds

# --------------------------------------------------
#               Workflow Summary Report
# --------------------------------------------------
# Total execution time: 309.73 seconds.
# Successfully processed: 10 papers.
# Failed to process:      0 papers.
# Total pages analyzed:   197.
# Average time per page:  5.69 seconds.
# --------------------------------------------------

# LOCAL MACOS
# --------------------------------------------------
#               Workflow Summary Report
# --------------------------------------------------
# Total execution time: 311.19 seconds.
# Successfully processed: 10 papers.
# Failed to process:      0 papers.
# Total pages analyzed:   197.
# Average time per page:  5.52 seconds.
# -------------------------------------------------
