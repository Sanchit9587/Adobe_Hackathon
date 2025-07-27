import json
from pathlib import Path
from collections import Counter

def process_section(section, all_headings):
    """
    Recursively extracts heading information (text, page, and level)
    from a section and its children.
    """
    if "heading" in section:
        heading_info = {
            "text": section["heading"]["text"].replace('\n', ' ').strip(), # Clean up newlines
            "page": section["heading"]["page"],
            "level": section["level"] # Use the 'level' attribute directly
        }
        all_headings.append(heading_info)
    if "children" in section:
        for child in section["children"]:
            process_section(child, all_headings)

def transform_pdfstructure_output(pdfstructure_output_data):
    """
    Transforms the pdfstructure JSON output into the desired file02.json format.
    It identifies the heading level (H1, H2, etc.) that has the most entries
    and reclassifies that specific level as 'body text'.

    Args:
        pdfstructure_output_data (dict): The loaded JSON data from pdfstructure's output.

    Returns:
        dict: A dictionary representing the data in the file02.json format.
    """
    all_headings_raw = []
    
    # Extract all headings and their levels by recursively traversing the 'elements'
    if "elements" in pdfstructure_output_data:
        for element in pdfstructure_output_data["elements"]:
            process_section(element, all_headings_raw)

    if not all_headings_raw:
        return {"title": "No Title Found", "outline": []}

    # Step 1: Perform initial classification (Level N -> H(N+1))
    # and collect counts of each H level
    initial_classified_headings = []
    h_level_counts = Counter()

    # Heuristic for document title:
    # If the very first extracted heading has level 0 and is on page 1,
    # consider it the main document title and exclude it from the outline list.
    document_title = "Document Outline" # Default title if not found explicitly
    headings_for_initial_classification = all_headings_raw

    if all_headings_raw and all_headings_raw[0]["level"] == 0 and all_headings_raw[0]["page"] == 1:
        document_title = all_headings_raw[0]["text"]
        headings_for_initial_classification = all_headings_raw[1:] # Skip this first element

    for heading in headings_for_initial_classification:
        # Classify based on the 'level' attribute: Level N maps to H(N+1)
        h_label = f"H{heading['level'] + 1}"
        initial_classified_headings.append({
            "level": h_label,
            "text": heading["text"],
            "page": heading["page"]
        })
        h_level_counts[h_label] += 1

    # Step 2: Find the H level with the most texts
    most_frequent_h_level = None
    if h_level_counts:
        # Find the H level with the highest count
        most_frequent_h_level = max(h_level_counts, key=h_level_counts.get)
        # Ensure 'body text' is not chosen if it somehow appears in counts (shouldn't with this logic)
        if most_frequent_h_level.startswith("H"):
            print(f"Identified '{most_frequent_h_level}' as the most frequent heading level to be reclassified as 'body text'.")
        else:
            most_frequent_h_level = None # Reset if it's not an H level

    # Step 3: Prepare the final outline, reclassifying the most frequent H level
    transformed_outline = []
    for heading in initial_classified_headings:
        current_level = heading["level"]
        if current_level == most_frequent_h_level:
            current_level = "body text" # Reclassify
        
        transformed_outline.append({
            "level": current_level,
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
    output_file_path = Path("content/formatted.json") # Save output in the same content folder

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

    # Save the transformed data to formatted.json
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_output, f, indent=4, ensure_ascii=False)

    print(f"Transformation complete. Output saved to '{output_file_path}'")
