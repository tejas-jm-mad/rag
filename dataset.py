import os
import shutil
import re

def organize_dataset(processed_markdown_dir, paper_metadata_dir, downloaded_pdfs_dir, output_dir):
    """
    Organizes research papers from three source directories into a clean dataset directory.

    This function iterates through metadata files, finds corresponding markdown and PDF files
    based on a unique paper ID, and copies them to a structured output directory if all
    three files exist.

    Args:
        processed_markdown_dir (str): Path to the directory with markdown files.
        paper_metadata_dir (str): Path to the directory with JSON metadata files.
        downloaded_pdfs_dir (str): Path to the directory with PDF files.
        output_dir (str): Path to the target directory where the dataset will be created.
    """
    # 1. Define destination paths
    dest_pdf_dir = os.path.join(output_dir, 'paper')
    dest_markdown_dir = os.path.join(output_dir, 'markdown')
    dest_metadata_dir = os.path.join(output_dir, 'api_metadata')

    # 2. Create destination directories if they don't exist
    print(f"Creating output directories in '{output_dir}'...")
    os.makedirs(dest_pdf_dir, exist_ok=True)
    os.makedirs(dest_markdown_dir, exist_ok=True)
    os.makedirs(dest_metadata_dir, exist_ok=True)
    print("Directories created successfully.\n")

    # --- More efficient approach: Index existing files first ---
    print("Indexing files from source directories...")
    # Regex to extract the unique ID (e.g., '1701.06538v1') from various filename formats
    id_pattern = re.compile(r"(\d{4}\.\d{5}v\d+)")

    def create_file_map(directory):
        """Creates a mapping from unique ID to full filepath."""
        file_map = {}
        for filename in os.listdir(directory):
            match = id_pattern.search(filename)
            if match:
                unique_id = match.group(1)
                file_map[unique_id] = os.path.join(directory, filename)
        return file_map

    markdown_map = create_file_map(processed_markdown_dir)
    pdf_map = create_file_map(downloaded_pdfs_dir)
    print(f"Found {len(markdown_map)} markdown files and {len(pdf_map)} PDF files to process.\n")

    # 3. Iterate through metadata files as the source of truth
    files_copied_count = 0
    files_skipped_count = 0
    
    metadata_files = os.listdir(paper_metadata_dir)
    total_metadata_files = len(metadata_files)
    print(f"Starting to process {total_metadata_files} metadata files...")

    for i, meta_filename in enumerate(metadata_files):
        if not meta_filename.endswith('.json'):
            continue

        match = id_pattern.search(meta_filename)
        if not match:
            print(f"  - WARNING: Could not extract ID from '{meta_filename}'. Skipping.")
            files_skipped_count += 1
            continue
        
        unique_id = match.group(1)

        # 4. Check if corresponding markdown and PDF files exist in our maps
        if unique_id in markdown_map and unique_id in pdf_map:
            # All three files exist, proceed with copying
            try:
                # Define source paths
                src_meta_path = os.path.join(paper_metadata_dir, meta_filename)
                src_markdown_path = markdown_map[unique_id]
                src_pdf_path = pdf_map[unique_id]

                # Define destination paths with clean names
                dest_meta_path = os.path.join(dest_metadata_dir, f"{unique_id}.json")
                dest_markdown_path = os.path.join(dest_markdown_dir, f"{unique_id}.md")
                dest_pdf_path = os.path.join(dest_pdf_dir, f"{unique_id}.pdf")
                
                # Copy the files
                shutil.copy2(src_meta_path, dest_meta_path)
                shutil.copy2(src_markdown_path, dest_markdown_path)
                shutil.copy2(src_pdf_path, dest_pdf_path)
                
                if (i + 1) % 50 == 0: # Print progress update every 50 files
                   print(f"  - Progress: {i+1}/{total_metadata_files} | Copied set for ID: {unique_id}")

                files_copied_count += 1
            except Exception as e:
                print(f"  - ERROR copying files for ID {unique_id}: {e}")
                files_skipped_count += 1
        else:
            # A corresponding file is missing, so skip this set
            missing = []
            if unique_id not in markdown_map:
                missing.append("markdown")
            if unique_id not in pdf_map:
                missing.append("PDF")
            # print(f"  - Skipping ID {unique_id}: Missing {', '.join(missing)} file(s).")
            files_skipped_count += 1

    print("\n---------------------------------")
    print("      Processing Complete      ")
    print("---------------------------------")
    print(f"Total metadata files found: {total_metadata_files}")
    print(f"Complete sets copied:       {files_copied_count}")
    print(f"File sets skipped:          {files_skipped_count}")
    print("---------------------------------")


if __name__ == '__main__':
    
    # --- Set your directory paths here ---
    PROCESSED_MARKDOWN_DIR = 'processed_markdown'
    PAPER_METADATA_DIR = 'paper_metadata'
    DOWNLOADED_PDFS_DIR = 'downloaded_pdfs'
    OUTPUT_DATASET_DIR = 'dataset'

    # Run the organization script
    organize_dataset(
        PROCESSED_MARKDOWN_DIR,
        PAPER_METADATA_DIR,
        DOWNLOADED_PDFS_DIR,
        OUTPUT_DATASET_DIR
    )
