import requests
import xml.etree.ElementTree as ET
import re
from pprint import pprint


def get_arxiv_details(url: str) -> dict:
    """
    Fetches metadata for an arXiv paper given its URL.
    Handles character encoding correctly.
    """
    match = re.search(r"(\d{4}\.\d{4,5})", url)
    if not match:
        return {"error": "Could not find a valid arXiv ID in the URL."}
    arxiv_id = match.group(1)

    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()

        # Use response.content to avoid encoding errors.
        # The XML parser will correctly handle the UTF-8 encoding from the raw bytes.
        root = ET.fromstring(response.content)

        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        entry = root.find("atom:entry", namespaces)
        if entry is None:
            return {"error": f"No entry found for arXiv ID {arxiv_id}."}

        # Helper function to safely get text, replacing newlines
        def get_text(element, tag):
            found = element.find(tag, namespaces)
            if found is not None and found.text:
                return found.text.strip().replace("\n", " ")
            return None

        paper_details = {
            "id": arxiv_id,
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


# --- Example Usage ---
paper_url = "https://arxiv.org/pdf/2505.13002"
details = get_arxiv_details(paper_url)

pprint(details)

import json

details = get_arxiv_details(paper_url)

# Write to a file with explicit UTF-8 encoding
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(details, f, ensure_ascii=False, indent=4)

print("Details saved to output.json")
