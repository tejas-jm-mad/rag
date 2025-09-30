import os
import time
import requests
import xml.etree.ElementTree as ET

# --- Configuration ---
# The path to your list of arXiv IDs.
IDS_FILE_PATH = "benchmark/selected_papers.txt"

# The top-level directory where you want to save the categorized PDFs.
OUTPUT_DIR = "benchmark/pdfs"
# ---------------------


def download_and_sort_arxiv_pdfs():
    """
    Reads arXiv IDs, fetches their primary category via the API,
    and downloads the PDFs into category-specific subdirectories.
    """
    # 1. Ensure the top-level output directory exists
    print(f"✅ Preparing output directory at: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Read the list of paper IDs
    try:
        with open(IDS_FILE_PATH, 'r', encoding='utf-8') as f:
            paper_ids = [line.strip() for line in f if line.strip()]
        print(f"🔎 Found {len(paper_ids)} paper IDs in '{IDS_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{IDS_FILE_PATH}' was not found.")
        return

    # 3. Loop through IDs, fetch metadata, and download each PDF
    print("\n🚀 Starting download process...")
    success_count = 0
    skipped_count = 0
    fail_count = 0
    
    # Define the XML namespaces used by the arXiv API response
    namespaces = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    for i, paper_id in enumerate(paper_ids):
        print(f"[{i+1}/{len(paper_ids)}] Processing ID: {paper_id}")
        
        try:
            # STEP A: Fetch metadata to find the primary_category
            meta_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
            meta_response = requests.get(meta_url)
            meta_response.raise_for_status() # Check for request errors

            # Parse the XML response
            root = ET.fromstring(meta_response.text)
            entry = root.find('atom:entry', namespaces)
            primary_category_element = entry.find('arxiv:primary_category', namespaces)
            primary_category = primary_category_element.get('term')

            print(f"  -> Metadata: Found primary category '{primary_category}'.")

            # STEP B: Prepare directories and file paths
            category_dir = os.path.join(OUTPUT_DIR, primary_category)
            os.makedirs(category_dir, exist_ok=True)
            output_path = os.path.join(category_dir, f"{paper_id}.pdf")

            # Check if the file already exists
            if os.path.exists(output_path):
                print("  -> Status: Already exists. Skipping.")
                skipped_count += 1
                # Wait before the next API call, even if skipping
                time.sleep(3)
                continue

            # STEP C: Download the actual PDF
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            pdf_response = requests.get(pdf_url, timeout=30)
            pdf_response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(pdf_response.content)

            print(f"  -> Status: Success! Saved to {output_path}")
            success_count += 1

        except requests.exceptions.RequestException as e:
            print(f"  -> Status: Failed for ID {paper_id}.")
            print(f"     Reason: {e}")
            fail_count += 1
        except (AttributeError, ET.ParseError) as e:
            print(f"  -> Status: Failed for ID {paper_id}.")
            print(f"     Reason: Could not parse metadata from API response. {e}")
            fail_count += 1
        
        # Per arXiv API guidelines, wait 3 seconds between requests
        print("  -> Waiting 3 seconds before next request...")
        time.sleep(3)


    # 4. Print a final summary
    print("\n🎉 Download process complete!")
    print(f"   Successfully downloaded: {success_count} files")
    print(f"   Skipped (already exist): {skipped_count} files")
    print(f"   Failed to process: {fail_count} files")


if __name__ == "__main__":
    download_and_sort_arxiv_pdfs()