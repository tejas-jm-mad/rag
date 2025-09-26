import requests
import xml.etree.ElementTree as ET
import re
import time
import json
import random
import itertools
from pathlib import Path
from datetime import datetime, timedelta
from docling.document_converter import DocumentConverter
from multiprocessing import Pool
import warnings
from pypdf import PdfReader
from pypdf.errors import PdfReadError

warnings.filterwarnings("ignore", message=".*pin_memory.*")

# ==============================================================================
# == CONFIGURATION - SET YOUR PARAMETERS HERE
# ==============================================================================
MAX_WORKERS = 5 
TARGET_PAPER_COUNT = 300
QUERY_CHUNK_SIZE = 20
BATCH_SIZE = 15 # The number of papers to process in each distinct batch

ARXIV_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.RO", "cs.NE",
    "stat.ML", "math.OC", "math.PR", "math.ST",
    "physics.data-an", "physics.comp-ph", "physics.flu-dyn",
    "q-bio.PE", "econ.EM"
]

DAYS_TO_SEARCH = 60 

PDF_SAVE_DIR = "downloaded_pdfs"
MD_SAVE_DIR = "processed_markdown"
METADATA_SAVE_DIR = "paper_metadata"

# ==============================================================================
# == SCRIPT LOGIC - NO NEED TO EDIT BELOW THIS LINE
# ==============================================================================

end_date_obj = datetime.now()
start_date_obj = end_date_obj - timedelta(days=DAYS_TO_SEARCH - 1)
START_DATE = start_date_obj.strftime("%Y-%m-%d")
END_DATE = end_date_obj.strftime("%Y-%m-%d")


def get_pdf_page_count(pdf_path):
    """Counts the number of pages in a PDF file with specific error handling."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except (PdfReadError, FileNotFoundError, TypeError) as e:
        print(f"[Warning] Could not read page count for {Path(pdf_path).name}: {e}")
        return 0

def _parse_entry_to_dict(entry, namespace):
    """Helper function to parse a single XML entry into a metadata dictionary."""
    def get_text(element, tag):
        found = element.find(tag, namespace)
        return found.text.strip().replace('\n', ' ') if found is not None and found.text else None
    
    paper_id_raw = get_text(entry, 'atom:id')
    paper_id = paper_id_raw.split('/abs/')[-1]
    
    pdf_link_element = entry.find('atom:link[@title="pdf"]', namespace)
    pdf_link = pdf_link_element.get('href') if pdf_link_element is not None else None
    
    details = {
        "id": paper_id, "versioned_id": paper_id_raw.split('/')[-1],
        "title": get_text(entry, 'atom:title'), "abstract": get_text(entry, 'atom:summary'),
        "authors": [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)],
        "published_date": get_text(entry, 'atom:published'), "updated_date": get_text(entry, 'atom:updated'),
        "primary_category": entry.find('arxiv:primary_category', namespace).get('term'),
        "categories": [cat.get('term') for cat in entry.findall('atom:category', namespace)],
        "pdf_link": pdf_link,
        "abstract_link": entry.find('atom:link[@rel="alternate"]', namespace).get('href'),
        "doi": get_text(entry, 'arxiv:doi'), "journal_ref": get_text(entry, 'arxiv:journal_ref')
    }
    return details

def search_arxiv_chunk(category, start_date, end_date, max_results):
    """Performs a single API call to get a chunk of papers."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d000000")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Ym%d235959")
    query = f'search_query=cat:{category}+AND+submittedDate:[{start_dt}+TO+{end_dt}]&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    base_url = 'http://export.arxiv.org/api/query?'
    try:
        time.sleep(3) 
        response = requests.get(base_url + query, timeout=20)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[Producer] Warning: API request for category '{category}' failed: {e}")
        return []
    root = ET.fromstring(response.content)
    namespace = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    return root.findall('atom:entry', namespace)

def find_papers_iteratively(target_count, categories, start_date, end_date):
    """A generator that finds unique papers by cycling through categories sequentially."""
    found_ids = set()
    papers_yielded = 0
    
    print(f"[Producer] Starting to find {target_count} unique papers...")

    while papers_yielded < target_count:
        print("[Producer] Starting a new sequential pass through the category list.")
        
        # --- MODIFIED SECTION: Loop sequentially instead of shuffling ---
        for category in categories:
            print(f"[Producer] Querying category '{category}' for a batch of papers...")
            entries = search_arxiv_chunk(category, start_date, end_date, QUERY_CHUNK_SIZE)
            
            if not entries:
                continue

            for entry in entries:
                metadata = _parse_entry_to_dict(entry, {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'})
                paper_id = metadata['id']
                
                if paper_id not in found_ids and metadata.get('pdf_link'):
                    found_ids.add(paper_id)
                    paper_id_sanitized = paper_id.replace('/', '_')
                    filename = f"arxiv{paper_id_sanitized}_metadata.json"
                    save_path = Path(METADATA_SAVE_DIR) / filename
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=4, ensure_ascii=False)
                    
                    yield {
                        "id": paper_id,
                        "title": metadata['title'],
                        "pdf_url": metadata['pdf_link']
                    }
                    papers_yielded += 1
                    if papers_yielded >= target_count:
                        print("[Producer] Target paper count reached. Stopping search.")
                        return

def sanitize_filename(name):
    """Removes characters that are invalid in filenames."""
    return re.sub(r'[\/:*?"<>|]', '_', name)

def download_pdf(paper_info, save_dir):
    """Downloads a PDF from a URL, validates it, and saves it to a directory."""
    filename = f"{paper_info['id'].replace('/', '_')}_{sanitize_filename(paper_info['title'])}.pdf"
    pdf_path = Path(save_dir) / filename
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(paper_info['pdf_url'], stream=True, headers=headers, timeout=60)
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '').lower()
    if 'application/pdf' not in content_type: return None
    first_chunk = next(response.iter_content(chunk_size=1024))
    if not first_chunk.startswith(b'%PDF-'): return None
    with open(pdf_path, 'wb') as f:
        f.write(first_chunk)
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return str(pdf_path)

def process_with_docling(pdf_path, save_dir):
    """Processes a PDF file with Docling and saves it as Markdown."""
    pdf_path_obj = Path(pdf_path)
    md_path = Path(save_dir) / (pdf_path_obj.stem + ".md")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown_content = result.document.export_to_markdown()
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    return str(md_path)

def download_and_process_paper(paper_info):
    """A single task that downloads, processes, and times a paper."""
    start_time = time.monotonic()
    paper_id = paper_info['id']
    
    try:
        pdf_path_str = download_pdf(paper_info, PDF_SAVE_DIR)
        
        if pdf_path_str:
            page_count = get_pdf_page_count(pdf_path_str)
            if page_count == 0:
                duration = time.monotonic() - start_time
                return {'status': 'download_failed_corrupt', 'paper_id': paper_id, 'duration': duration, 'pages': 0}
            
            md_path_str = process_with_docling(pdf_path_str, MD_SAVE_DIR)
            duration = time.monotonic() - start_time

            if md_path_str:
                return {'status': 'success', 'paper_id': paper_id, 'duration': duration, 'pages': page_count}
            else:
                return {'status': 'processing_failed', 'paper_id': paper_id, 'duration': duration, 'pages': page_count}
        else:
            duration = time.monotonic() - start_time
            return {'status': 'download_failed', 'paper_id': paper_id, 'duration': duration, 'pages': 0}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        duration = time.monotonic() - start_time
        return {'status': 'task_exception', 'paper_id': paper_id, 'duration': duration, 'error': str(e)}


def main():
    """Main function to orchestrate the batched parallel workflow."""
    workflow_start_time = time.monotonic()
    print("--- arXiv Paper Processing Workflow Initialized ---")
    
    Path(PDF_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(MD_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(METADATA_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    total_pages_processed = 0
    total_time_spent_processing = 0.0

    paper_generator = find_papers_iteratively(TARGET_PAPER_COUNT, ARXIV_CATEGORIES, START_DATE, END_DATE)
    
    with Pool(processes=MAX_WORKERS, maxtasksperchild=1) as pool:
        while True:
            print(f"\n[Main] Fetching a new batch of up to {BATCH_SIZE} papers...")
            batch = list(itertools.islice(paper_generator, BATCH_SIZE))
            
            if not batch:
                print("[Main] Producer has no more papers. Workflow finishing.")
                break

            print(f"[Main] Submitting batch of {len(batch)} papers to workers...")

            results = pool.map(download_and_process_paper, batch)

            print(f"[Main] Processing results for the completed batch...")
            for result in results:
                if not result: continue

                paper_id = result['paper_id']
                duration = result['duration']
                
                if result['status'] == 'success':
                    success_count += 1
                    pages = result['pages']
                    total_pages_processed += pages
                    total_time_spent_processing += duration
                    print(f"Success: Paper {paper_id} ({pages} pages) processed in {duration:.2f} seconds.")
                else:
                    failure_count += 1
                    status_message = result['status'].replace('_', ' ').capitalize()
                    error_message = result.get('error', 'N/A')
                    print(f"Failure: {status_message} for paper {paper_id}. Took {duration:.2f} seconds. Reason: {error_message}")
    
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