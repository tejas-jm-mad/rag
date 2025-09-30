import os
from lxml import etree
import logging

# --- Configuration ---
# Set up the input and output directories. Modify these paths as needed.
INPUT_DIRECTORY = "dataset/grobid_xml"
OUTPUT_DIRECTORY = "dataset/grobid_text"
# --- End Configuration ---

# Set up basic logging to see progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the TEI namespace used by GROBID to correctly find elements
NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

def get_text_recursive(element):
    """
    Recursively extracts and cleans text from an element and its children.
    It joins text parts, handles special tags, and normalizes whitespace.
    """
    if element is None:
        return ""
    
    # List to hold all text parts from the element and its children
    text_parts = []

    # Include the element's own text (text before the first child)
    if element.text:
        text_parts.append(element.text)

    # Process all child elements
    for child in element:
        # Special formatting for inline citations
        if child.tag == f"{{{NS['tei']}}}ref" and child.get('type') == 'bibr':
            text_parts.append(f"[{child.text.strip()}]")
        # Recursively get text from other children
        else:
            text_parts.append(get_text_recursive(child))
        
        # Important: Include text that follows a child element (its 'tail')
        if child.tail:
            text_parts.append(child.tail)
            
    # Join all parts and clean up whitespace for a clean output
    return ' '.join("".join(text_parts).split())

def format_table(table_element):
    """Formats a <table> element into a simple, readable text representation."""
    if table_element is None:
        return ""
    
    table_text = ["\n--- TABLE START ---"]
    for row in table_element.findall('.//tei:row', namespaces=NS):
        # Extract text from each cell and join with a separator
        row_cells = [get_text_recursive(cell) for cell in row.findall('tei:cell', namespaces=NS)]
        table_text.append(" | ".join(row_cells))
    table_text.append("--- TABLE END ---\n")
    return "\n".join(table_text)

def convert_xml_to_text(xml_path):
    """
    Main function to parse a single GROBID TEI XML file and convert it to a
    structured, formatted plain text string.
    """
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        
        output_content = []

        # 1. Title
        title = root.find('.//tei:titleStmt/tei:title', namespaces=NS)
        if title is not None:
            title_text = get_text_recursive(title)
            output_content.append(title_text.upper())
            output_content.append("=" * len(title_text))

        # 2. Authors
        authors = root.findall('.//tei:author/tei:persName', namespaces=NS)
        author_names = [get_text_recursive(name) for name in authors]
        if author_names:
            output_content.append(f"Authors: {', '.join(author_names)}\n")

        # 3. Abstract
        abstract = root.find('.//tei:abstract//tei:p', namespaces=NS)
        if abstract is not None:
            output_content.append("## ABSTRACT")
            output_content.append(get_text_recursive(abstract) + "\n")

        # 4. Main Body Content
        body = root.find('.//tei:body', namespaces=NS)
        if body is not None:
            # Iterate through all direct children of the body
            for element in body.iterchildren():
                # Handle sections (<div> elements)
                if element.tag == f"{{{NS['tei']}}}div":
                    heading = element.find('tei:head', namespaces=NS)
                    if heading is not None:
                        output_content.append(f"\n## {get_text_recursive(heading).upper()}\n")
                    
                    # Process paragraphs, formulas, and tables within the section
                    for sub_element in element.iterchildren():
                        if sub_element.tag == f"{{{NS['tei']}}}p":
                            output_content.append(get_text_recursive(sub_element) + "\n")
                        elif sub_element.tag == f"{{{NS['tei']}}}formula":
                            output_content.append(f"    [FORMULA]: {get_text_recursive(sub_element)}\n")
                        elif sub_element.tag == f"{{{NS['tei']}}}figure" and sub_element.find('tei:table', namespaces=NS) is not None:
                             output_content.append(format_table(sub_element.find('tei:table', namespaces=NS)))

        # 5. References
        bibl_structs = root.findall('.//tei:listBibl/tei:biblStruct', namespaces=NS)
        if bibl_structs:
            output_content.append("\n" + "="*40)
            output_content.append("## REFERENCES\n")
            for i, item in enumerate(bibl_structs, 1):
                output_content.append(f"[{i}] {get_text_recursive(item)}")
        
        return "\n".join(output_content)

    except Exception as e:
        logging.error(f"Could not process file {xml_path}. Error: {e}")
        return None

def main():
    """
    Main function to orchestrate the batch conversion process.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    logging.info(f"Starting batch conversion...")
    logging.info(f"Input directory: '{INPUT_DIRECTORY}'")
    logging.info(f"Output directory: '{OUTPUT_DIRECTORY}'")

    # Check if the input directory exists
    if not os.path.isdir(INPUT_DIRECTORY):
        logging.error(f"Input directory '{INPUT_DIRECTORY}' not found. Please create it and add your XML files.")
        return

    # Process each file in the input directory
    file_list = [f for f in os.listdir(INPUT_DIRECTORY) if f.endswith('.xml')]
    if not file_list:
        logging.warning(f"No .xml files found in '{INPUT_DIRECTORY}'.")
        return

    for filename in file_list:
        input_path = os.path.join(INPUT_DIRECTORY, filename)
        
        # Create the corresponding output filename with a .txt extension
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
        
        logging.info(f"Converting '{filename}' -> '{output_filename}'")
        
        # Convert the XML to a text string
        formatted_text = convert_xml_to_text(input_path)
        
        # Write the result to the output file
        if formatted_text:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
                
    logging.info("Batch conversion complete.")


if __name__ == "__main__":
    main()