import os
import time
import requests

# --- Configuration ---
# The path to your list of arXiv IDs.
IDS_FILE_PATH = "benchmark/selected_papers.txt"

# The directory where you want to save the downloaded PDFs.
OUTPUT_DIR = "benchmark/pdfs"
# ---------------------


def download_arxiv_pdfs():
    """
    Reads a list of arXiv IDs from a text file and downloads the
    corresponding PDF for each ID.
    """
    # 1. Ensure the output directory exists
    print(f"✅ Preparing output directory at: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Read the list of paper IDs from the text file
    try:
        with open(IDS_FILE_PATH, 'r', encoding='utf-8') as f:
            # Read all lines, stripping whitespace and removing any empty lines
            paper_ids = [line.strip() for line in f if line.strip()]
        print(f"🔎 Found {len(paper_ids)} paper IDs in '{IDS_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{IDS_FILE_PATH}' was not found. Please check the path.")
        return

    # 3. Loop through IDs and download each PDF
    print("\n🚀 Starting download process...")
    success_count = 0
    skipped_count = 0
    fail_count = 0

    for i, paper_id in enumerate(paper_ids):
        # The direct PDF download URL for an arXiv paper
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        output_path = os.path.join(OUTPUT_DIR, f"{paper_id}.pdf")
        
        print(f"[{i+1}/{len(paper_ids)}] Processing ID: {paper_id}")

        # Check if the file already exists to allow for resuming
        if os.path.exists(output_path):
            print("  -> Status: Already exists. Skipping.")
            skipped_count += 1
            continue

        try:
            # Make the HTTP request to download the PDF
            response = requests.get(pdf_url, timeout=30)
            
            # Raise an exception for bad status codes (like 404 Not Found)
            response.raise_for_status()

            # Save the PDF content to a file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"  -> Status: Success! Saved to {output_path}")
            success_count += 1

        except requests.exceptions.RequestException as e:
            print(f"  -> Status: Failed. Could not download from {pdf_url}")
            print(f"     Reason: {e}")
            fail_count += 1
        
        # Be a polite user and wait a moment between requests
        time.sleep(1)

    # 4. Print a final summary
    print("\n🎉 Download process complete!")
    print(f"   Successfully downloaded: {success_count} files")
    print(f"   Skipped (already exist): {skipped_count} files")
    print(f"   Failed to download: {fail_count} files")


if __name__ == "__main__":
    download_arxiv_pdfs()