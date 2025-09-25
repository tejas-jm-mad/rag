import os
import json
import spacy
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Configuration ---
# Define input and output directories as global constants
INPUT_DIRECTORY = "arxiv_cleaned"
OUTPUT_DIRECTORY = "arxiv_preprocessed"

# --- Worker Process Initialization ---
# This global variable will hold the spaCy model in each worker process.
nlp = None


def init_worker():
    """
    Initializer for each worker process. Loads the spaCy model once per process
    and adds the necessary 'sentencizer' component.
    """
    global nlp
    print("Initializing a worker process...")
    # Load the model with parser and NER disabled for speed
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # Explicitly add the sentencizer component to the pipeline
    nlp.add_pipe("sentencizer")


# --- Worker Function ---
def process_file(filename):
    """
    This function processes a single JSON file. It reads the data, splits the
    description into sentences, and writes the result to a new file.
    """
    if not filename.endswith(".json"):
        return None  # Skip non-JSON files

    input_file_path = os.path.join(INPUT_DIRECTORY, filename)
    output_file_path = os.path.join(OUTPUT_DIRECTORY, filename)

    try:
        with open(input_file_path, "r", encoding="utf-8") as file:
            original_info = json.load(file)

        paper_text = original_info.get("original_description")
        if not paper_text:
            return (filename, "Skipped", "Missing 'original_description' key")

        # Clean the text
        paper_text = paper_text.replace("\n", " ")

        paper_abstract = original_info.get("abstract")
        if not paper_abstract:
            return (filename, "Skipped", "Missing 'abstract' key")

        # Process with the globally loaded nlp model
        description_doc = nlp(paper_text)
        description_sentences = [sent.text for sent in description_doc.sents]
        num_description_sentences = len(description_sentences)

        # Process with the globally loaded nlp model
        abstract_doc = nlp(paper_abstract)
        abstract_sentences = [sent.text for sent in abstract_doc.sents]
        num_abstract_sentences = len(abstract_sentences)

        abstract_description_combined_sentences = (
            abstract_sentences + description_sentences
        )
        abstract_description_combined_sentences_number = len(
            abstract_description_combined_sentences
        )

        original_info["Description_Sentences"] = description_sentences
        original_info["Description_Sentences_Number"] = num_description_sentences

        original_info["Abstract_Sentences"] = abstract_sentences
        original_info["Abstract_Sentences_Number"] = num_abstract_sentences

        original_info["Abstract_Description_Combined_Sentences"] = (
            abstract_description_combined_sentences
        )
        original_info["Abstract_Description_Combined_Sentences_Number"] = (
            abstract_description_combined_sentences_number
        )

        # Write the modified dictionary to the new file
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            json.dump(original_info, outfile, indent=4)

        return (filename, "Success", num_description_sentences)

    except Exception as e:
        # Return the filename and error for tracking
        return (filename, "Failed", str(e))


# --- Main Execution Block ---
if __name__ == "__main__":
    start_time = time.perf_counter()

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    if not os.path.isdir(INPUT_DIRECTORY):
        print(f"Error: Input directory '{INPUT_DIRECTORY}' not found.")
    else:
        file_list = sorted(os.listdir(INPUT_DIRECTORY))

        # Determine the number of processes to use
        num_processes = cpu_count()
        print(
            f"Starting parallel processing on {num_processes} cores for {len(file_list)} files..."
        )

        success_count = 0
        failed_count = 0

        # Create a pool of worker processes
        with Pool(processes=num_processes, initializer=init_worker) as pool:
            # Use tqdm to create a progress bar
            # imap_unordered is efficient as it processes results as they complete
            for result in tqdm(
                pool.imap_unordered(process_file, file_list), total=len(file_list)
            ):
                if result is None:
                    continue

                filename, status, message = result
                if status == "Success":
                    success_count += 1
                else:
                    failed_count += 1
                    print(f"Error processing {filename}: {message}")

        print(f"\nFinished processing.")
        print(f"Total files successfully processed: {success_count}")
        print(f"Total files failed: {failed_count}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
