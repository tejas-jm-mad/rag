import os 
import json
import spacy
import time

start_time = time.perf_counter()
# Define input and output directories
input_directory = 'arxiv_cleaned'
output_directory = 'arxiv_preprocessed'
counter = 0
nlp = spacy.load("en_core_web_sm")

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Check if the input directory exists
if not os.path.isdir(input_directory):
    print(f"Error: Input directory '{input_directory}' not found.")
else:
    file_list = sorted(os.listdir(input_directory)) # Sort for predictable order
    for filename in file_list:
        if not filename.endswith('.json'): # Skip non-JSON files
            continue
            
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)
        # print(f"Processing: {input_file_path}")

        try:
            with open(input_file_path, 'r', encoding='utf-8') as file:
                original_info = json.load(file)

            # Extract the URL from the original file's data
            paper_text = original_info.get("original_description")
            paper_text = paper_text.replace("\n", " ")
            if not paper_text:
                print(f"Error: Missing key 'original_description' in {filename}. Skipping.")
                continue
            
            doc = nlp(paper_text)
            # The doc.sents generator contains the detected sentences
            sentences = [sent.text for sent in doc.sents]

            original_info["Description_Sentences"] = sentences
            
            original_info["Description_Sentences_Number"] = len(sentences)

            # Write the modified dictionary to the new file
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                json.dump(original_info, outfile, indent=4) 
            
            print(f"FileName : {filename}")
            print(f"OriginalDescription Length : {len(paper_text)}")
            print(f"Number of Sentences : {len(sentences)}")

        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {filename}. Check for malformed JSON.")
        except Exception as e:
            # A general catch-all for other unexpected errors
            print(f"An unexpected error occurred with {filename}: {e}")
        counter += 1

print(f"\nFinished processing. Total files successfully processed: {counter}")
end_time = time.perf_counter()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.6f} seconds")
