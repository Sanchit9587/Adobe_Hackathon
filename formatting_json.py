import json
import collections
from pathlib import Path

def process_section(section, all_headings):
    """Recursively extracts heading information from a section and its children."""
    if "heading" in section:
        heading_info = {
            "text": section["heading"]["text"].replace('\n', ' ').strip(), # Clean up newlines
            "page": section["heading"]["page"],
            "mean_size": section["heading"]["style"]["mean_size"]
        }
        all_headings.append(heading_info)
    if "children" in section:
        for child in section["children"]:
            process_section(child, all_headings)

def transform_pdfstructure_output(pdfstructure_output_data):
    """
    Transforms the pdfstructure JSON output into the desired file02.json format
    with the updated heading classification rules (H1-H5, rest body text).

    Args:
        pdfstructure_output_data (dict): The loaded JSON data from pdfstructure's output.

    Returns:
        dict: A dictionary representing the data in the file02.json format.
    """
    all_headings_raw = []
    
    # Extract all headings and their mean_sizes by recursively traversing the 'elements'
    if "elements" in pdfstructure_output_data:
        for element in pdfstructure_output_data["elements"]:
            process_section(element, all_headings_raw)

    if not all_headings_raw:
        return {"title": "No Title Found", "outline": []}

    # Identify unique font sizes and sort them in descending order
    unique_mean_sizes = sorted(list(set(h["mean_size"] for h in all_headings_raw)), reverse=True)

    # Determine the mapping from mean_size to H1, H2, ... H5 or 'body text'
    size_to_level_map = {}
    for i, size in enumerate(unique_mean_sizes):
        if i < 5: # Assign H1 to H5 for the top 5 largest distinct font sizes
            size_to_level_map[size] = f"H{i + 1}"
        else: # Everything after H5 (6th largest distinct size and smaller) is 'body text'
            size_to_level_map[size] = "body text"

    # Prepare the final outline
    transformed_outline = []
    document_title = "Document Outline" # Default title if not found explicitly

    # Heuristic to identify the document title:
    # Assume the very first heading with the largest font size (mapped to H1) and on page 1 is the title.
    # This also removes it from the outline if it's considered the main title.
    headings_for_outline = all_headings_raw
    if all_headings_raw and all_headings_raw[0]["mean_size"] == unique_mean_sizes[0] and all_headings_raw[0]["page"] == 1:
        document_title = all_headings_raw[0]["text"]
        # Skip this first element as it's now the title
        headings_for_outline = all_headings_raw[1:]
            
    for heading in headings_for_outline:
        level_str = size_to_level_map.get(heading["mean_size"], "body text")
        
        transformed_outline.append({
            "level": level_str,
            "text": heading["text"],
            "page": heading["page"]
        })
    
    return {
        "title": document_title,
        "outline": transformed_outline
    }

# --- Main execution ---
if __name__ == "__main__":
    # Define paths relative to the script's execution location (project root)
    # This assumes you run this script from the root directory of your project.
    input_file_path = Path("content/extracted.json")
    output_file_path = Path("content/file02_output.json") # Save output in the same content folder

    # Load the pdfstructure output JSON (from extracted.json)
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            pdfstructure_output_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{input_file_path}' not found. Please ensure the file exists at this path when running the script.")
        print("Tip: Run this script from the directory containing 'content/' folder.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode '{input_file_path}'. Invalid JSON format.")
        exit()

    # Transform the data
    transformed_output = transform_pdfstructure_output(pdfstructure_output_data)

    # Save the transformed data to file02_output.json
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_output, f, indent=4, ensure_ascii=False)

    print(f"Transformation complete. Output saved to '{output_file_path}'")