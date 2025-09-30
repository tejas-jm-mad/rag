import os
import json
import shutil
import pandas as pd

# --- Configuration ---
API_METADATA_DIR = "dataset/api_metadata"
PAPER_DIR = "dataset/paper"
BENCHMARK_DIR = "dataset/benchmark"

# Define specific output directories for PDFs and JSONs
PDF_OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "pdfs")
JSON_OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "json")

SAMPLE_SIZE = 25
# Use a fixed random state for reproducible results
RANDOM_STATE = 42
# ---------------------


def create_structured_benchmark():
    """
    Creates a stratified sample of papers, copying their PDFs and JSON metadata
    into separate, category-structured output directories, and saves a list
    of the selected paper IDs.
    """
    # 1. Create the output directories
    print(f"✅ Preparing output directories inside: {BENCHMARK_DIR}")
    os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

    # 2. Read metadata
    print(f"📊 Analyzing metadata from: {API_METADATA_DIR}")
    all_metadata = []
    json_files = [f for f in os.listdir(API_METADATA_DIR) if f.endswith('.json')]

    for file_name in json_files:
        file_path = os.path.join(API_METADATA_DIR, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'id' in data and 'primary_category' in data:
                    all_metadata.append({
                        'id': data['id'],
                        'primary_category': data['primary_category'],
                        'filename': file_name
                    })
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            print(f"  [!] Warning: Skipping file {file_name} due to error: {e}")

    if not all_metadata:
        print("❌ Error: No valid metadata found. Exiting.")
        return

    df = pd.DataFrame(all_metadata)

    print("\n--- Full Dataset Distribution ---")
    primary_category_counts = df['primary_category'].value_counts()
    print(primary_category_counts)
    print("---------------------------------\n")

    # 3. Perform stratified sampling
    print(f"🎯 Selecting a representative sample of {SAMPLE_SIZE} items...")
    proportions = df['primary_category'].value_counts(normalize=True)
    n_samples_per_cat = (proportions * SAMPLE_SIZE).round().astype(int)
    
    diff = SAMPLE_SIZE - n_samples_per_cat.sum()
    if diff != 0:
        largest_cat = n_samples_per_cat.idxmax()
        n_samples_per_cat[largest_cat] += diff
        print(f"   (Adjusted count for '{largest_cat}' by {diff} to meet total sample size)")

    print("\n--- Files to Sample per Category ---")
    print(n_samples_per_cat[n_samples_per_cat > 0])
    print("------------------------------------\n")

    final_sample_df = pd.DataFrame()
    for primary_cat, n_to_sample in n_samples_per_cat.items():
        if n_to_sample > 0:
            category_group = df[df['primary_category'] == primary_cat]
            actual_sample_size = min(int(n_to_sample), len(category_group))
            sampled_group = category_group.sample(n=actual_sample_size, random_state=RANDOM_STATE)
            final_sample_df = pd.concat([final_sample_df, sampled_group])

    # 4. NEW: Save the list of selected IDs to a text file
    ids_file_path = os.path.join(BENCHMARK_DIR, "selected_ids.txt")
    print(f"📝 Saving the list of {len(final_sample_df)} selected IDs to: {ids_file_path}")
    
    sampled_ids = final_sample_df['id'].tolist()
    with open(ids_file_path, 'w', encoding='utf-8') as f:
        # Join the list of IDs with newline characters and write in one go
        f.write('\n'.join(sampled_ids))

    # 5. Copy both the PDF and the corresponding JSON to the new structure
    print(f"📋 Copying {len(final_sample_df)} PDF and JSON files into respective subdirectories...")
    copied_pdf_count = 0
    copied_json_count = 0
    skipped_pdf_count = 0
    
    for index, row in final_sample_df.iterrows():
        paper_id = row['id']
        primary_category = row['primary_category']
        json_filename = row['filename']

        # Handle PDF Copying
        source_pdf_path = os.path.join(PAPER_DIR, f"{paper_id}.pdf")
        dest_pdf_category_dir = os.path.join(PDF_OUTPUT_DIR, primary_category)
        os.makedirs(dest_pdf_category_dir, exist_ok=True)
        dest_pdf_path = os.path.join(dest_pdf_category_dir, f"{paper_id}.pdf")

        if os.path.exists(source_pdf_path):
            shutil.copy2(source_pdf_path, dest_pdf_path)
            copied_pdf_count += 1
        else:
            print(f"  [!] Warning: Source PDF for id '{paper_id}' not found and was skipped: {source_pdf_path}")
            skipped_pdf_count += 1
        
        # Handle JSON Copying
        source_json_path = os.path.join(API_METADATA_DIR, json_filename)
        dest_json_category_dir = os.path.join(JSON_OUTPUT_DIR, primary_category)
        os.makedirs(dest_json_category_dir, exist_ok=True)
        dest_json_path = os.path.join(dest_json_category_dir, json_filename)

        if os.path.exists(source_json_path):
            shutil.copy2(source_json_path, dest_json_path)
            copied_json_count += 1
        else:
            print(f"  [!] Warning: Source JSON '{json_filename}' unexpectedly not found.")

    print("\n🎉 Process Complete!")
    print(f"   Successfully copied: {copied_pdf_count} PDFs to '{PDF_OUTPUT_DIR}'")
    print(f"   Successfully copied: {copied_json_count} JSONs to '{JSON_OUTPUT_DIR}'")
    print(f"   List of selected IDs saved to: '{ids_file_path}'")
    if skipped_pdf_count > 0:
        print(f"   Skipped (PDFs not found): {skipped_pdf_count} files")


if __name__ == "__main__":
    create_structured_benchmark()