import pdfplumber
import io
from PIL import Image

def process_pdf_with_plumber(pdf_bytes: bytes, paper_id: str):
    """
    Processes a PDF using pdfplumber to extract two-column text, tables, and images.

    Args:
        pdf_bytes: The content of the PDF file as bytes.
        paper_id: A unique ID for the paper to name image files.

    Returns:
        A dictionary containing the extracted text, tables, and image data.
    """
    extracted_data = {
        "full_text": "",
        "tables": [],
        "images": []
    }
    
    # Open the PDF from in-memory bytes
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # Loop through each page of the PDF
        for page_num, page in enumerate(pdf.pages):
            
            # 📄 1. Extract Text (handles two-column layouts automatically)
            # pdfplumber's default text extraction is layout-aware
            page_text = page.extract_text(use_text_flow= True)
            if page_text:
                extracted_data["full_text"] += page_text + "\n\n" # Add page break

            # 📊 2. Extract Tables
            page_tables = page.extract_tables()
            if page_tables:
                # Add page number for context
                extracted_data["tables"].extend([(page_num + 1, table) for table in page_tables])

            # 🖼️ 3. Extract Images
            for i, img_obj in enumerate(page.images):
                # Get the bounding box of the image
                bbox = (img_obj["x0"], img_obj["top"], img_obj["x1"], img_obj["bottom"])
                
                # Crop the page to the image's bounding box and render it
                cropped_page = page.crop(bbox)
                img = cropped_page.to_image(resolution=150)
                
                # Save the image to an in-memory buffer
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                
                # Create a unique filename for the image
                img_filename = f"{paper_id}_page{page_num+1}_img{i+1}.png"
                
                extracted_data["images"].append({
                    "filename": img_filename,
                    "data": img_buffer.getvalue()
                })
    
    return extracted_data

import feedparser
import requests
import time
import os
from pathlib import Path

def scrape_arxiv_category(category, max_results=4, download_dir="arxiv_output"):
    """
    Scrapes arXiv, processes PDFs with pdfplumber, and saves text, tables, and images.
    """
    # --- Setup Directories ---
    output_path = Path(download_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    
    # --- Scrape ArXiv ---
    base_url = 'http://export.arxiv.org/api/query?'
    # query = f'search_query=cat:{category}&start=0&max_results={max_results}'
    query = f'search_query=cat:{category}&sortBy=submittedDate&start=0&max_results={max_results}'
    response = feedparser.parse(base_url + query)

    all_papers_data = []

    for entry in response.entries:
        # ✅ FIX: Sanitize the paper ID by replacing both '.' and '/' with '_'
        paper_id = entry.id.split('/abs/')[-1].replace('.', '_').replace('/', '_') 
        
        print(f"\nProcessing paper: {entry.title} ({entry.id.split('/abs/')[-1]})") # Print original ID for clarity
        
        pdf_link = next((link.href for link in entry.links if link.get('title') == 'pdf'), None)

        paper_details = {
            'arxiv_id': paper_id,
            'title': entry.title,
            'abstract': entry.summary.replace('\n', ' '),
            'pdf_link': pdf_link,
            'full_text': "Could not retrieve text.",
            'tables': [],
            'images': []
        }

        if pdf_link:
            try:
                print("  -> Downloading PDF for processing...")
                pdf_response = requests.get(pdf_link, timeout=20)
                pdf_response.raise_for_status()
                
                # Use our powerful pdfplumber function
                extracted_content = process_pdf_with_plumber(pdf_response.content, paper_id)
                
                paper_details.update(extracted_content)
                print(f"  -> Extracted: {len(paper_details['tables'])} tables and {len(paper_details['images'])} images.")

                # This part will now work correctly
                for image in paper_details['images']:
                    img_save_path = images_path / image['filename']
                    with open(img_save_path, 'wb') as f:
                        f.write(image['data'])
                print(f"  -> Saved images to '{images_path.resolve()}'")

            except Exception as e:
                print(f"  -> Failed to process PDF: {e}")
        else:
            print("  -> No PDF link found.")
            
        all_papers_data.append(paper_details)
        time.sleep(3)
        
    print(f"\n✅ Finished scraping. Found {len(all_papers_data)} papers.")
    return all_papers_data

# --- Example Usage ---
import pprint

papers = scrape_arxiv_category('cs.AI', max_results=1) # Computer Vision papers often have images/tables

if papers:
    first_paper = papers[0]
    print("\n--- Details for the first paper ---")
    
    # Print the abstract
    print(f"Title: {first_paper['title']}")
    
    # Show the first 500 characters of the text
    print("\n--- Start of Extracted Text ---")
    print(first_paper['full_text'] + "...")
    
    # # Show one of the extracted tables, if any
    # if first_paper['tables']:
    #     print("\n--- Example of Extracted Table (from page", first_paper['tables'][0][0], ") ---")
    #     pprint.pprint(first_paper['tables'][0][1]) # The table data is the 2nd element
        
    print(f"\nImages have been saved to the 'arxiv_output/images' directory.")