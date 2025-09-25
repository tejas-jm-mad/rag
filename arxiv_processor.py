import os
import json
import requests
import xml.etree.ElementTree as ET
import re
import time


def get_arxiv_details(url: str) -> dict:
    """
    Fetches metadata for an arXiv paper given its URL.
    Handles character encoding correctly.
    """
    # Use regex to find the arXiv ID in the URL
    match = re.search(r"(\d{4}\.\d{4,5})", url)
    if not match:
        return {"error": "Could not find a valid arXiv ID in the URL."}
    arxiv_id = match.group(1)

    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

        # Use response.content (bytes) to let the XML parser handle UTF-8 encoding
        root = ET.fromstring(response.content)

        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        entry = root.find("atom:entry", namespaces)
        if entry is None:
            return {"error": f"No entry found for arXiv ID {arxiv_id}."}

        # Helper function to safely get text from an element and clean it
        def get_text(element, tag):
            found = element.find(tag, namespaces)
            if found is not None and found.text:
                return found.text.strip().replace("\n", " ")
            return None

        # Extract all details from the API response
        paper_details = {
            "arxiv_id": arxiv_id,
            "title": get_text(entry, "atom:title"),
            "abstract": get_text(entry, "atom:summary"),
            "authors": [
                author.find("atom:name", namespaces).text
                for author in entry.findall("atom:author", namespaces)
            ],
            "published_date": get_text(entry, "atom:published"),
            "updated_date": get_text(entry, "atom:updated"),
            "primary_category": entry.find("arxiv:primary_category", namespaces).get(
                "term"
            ),
            "categories": [
                cat.get("term") for cat in entry.findall("atom:category", namespaces)
            ],
            "pdf_link": entry.find('atom:link[@title="pdf"]', namespaces).get("href"),
            "abstract_link": entry.find('atom:link[@rel="alternate"]', namespaces).get(
                "href"
            ),
            "doi": get_text(entry, "arxiv:doi"),
            "journal_ref": get_text(entry, "arxiv:journal_ref"),
        }

        return paper_details

    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}
    except ET.ParseError:
        return {"error": "Failed to parse the XML response from arXiv."}


# --- Main Processing Logic ---

# Define input and output directories
input_directory = "arxiv"
output_directory = "arxiv_cleaned"
counter = 0

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Check if the input directory exists
if not os.path.isdir(input_directory):
    print(f"Error: Input directory '{input_directory}' not found.")
else:
    file_list = sorted(os.listdir(input_directory))  # Sort for predictable order
    for filename in file_list:
        if not filename.endswith(".json"):  # Skip non-JSON files
            continue

        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)
        print(f"Processing: {input_file_path}")

        try:
            with open(input_file_path, "r", encoding="utf-8") as file:
                original_info = json.load(file)

            # Extract the URL from the original file's data
            website_url = original_info.get("website_url")
            if not website_url:
                print(f"Error: Missing key 'website_url' in {filename}. Skipping.")
                continue

            # Fetch details from the arXiv API
            print(f" -> Fetching details for {website_url}")
            arxiv_details = get_arxiv_details(website_url)

            # Check if the API call returned an error
            if "error" in arxiv_details:
                print(
                    f" -> Error fetching details for {filename}: {arxiv_details['error']}. Skipping."
                )
                continue

            # Create the new flattened dictionary structure
            # It starts with everything from the API call...
            # ...and adds the two specific fields from the original file.
            # Using .get() provides default None values if keys are missing.
            flat_data = {
                **arxiv_details,
                "original_category": original_info.get("category"),
                "original_description": original_info.get("data", {}).get(
                    "description"
                ),
            }

            # Save the new flat data to the output directory
            with open(output_file_path, "w", encoding="utf-8") as outfile:
                json.dump(flat_data, outfile, indent=4, ensure_ascii=False)

            counter += 1
            print(
                f" -> Success! Saved cleaned file to: {output_file_path}. Count: {counter}"
            )

            # Wait for 1 seconds before the next API call
            print(" -> Waiting 1 seconds...")
            time.sleep(1)

        except FileNotFoundError:
            print(f"Error: {input_file_path} not found.")
        except json.JSONDecodeError:
            print(
                f"Error: Failed to decode JSON from {filename}. Check for malformed JSON."
            )
        except KeyError as e:
            print(f"Error: Missing key {e} in {filename}.")
        except TypeError:
            print(f"Error: Unexpected data type in {filename}. Expected a dictionary.")

print(f"\nFinished processing. Total files successfully processed: {counter}")
