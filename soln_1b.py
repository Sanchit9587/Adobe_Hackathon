import json
import math
import requests # Used for communicating with the local Ollama server
import datetime
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# ==============================================================================
# OFFLINE VECTOR GENERATION (REAL IMPLEMENTATION)
# ==============================================================================
# This is the actual implementation for the competition.
# It assumes you have downloaded the model and placed it in a folder
# named 'local-model' in the same directory as this script.

# 1. Load the model from the local directory. This is done only once.
print("Loading sentence-transformer model from './local-model'...")
try:
    model = SentenceTransformer('./local-model')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading sentence-transformer model: {e}")
    print("Please ensure the 'local-model' directory exists and contains the model files.")
    # Fallback to a placeholder if the model fails to load, to prevent crashing.
    # This part will not produce meaningful results but allows the script to run.
    model = None

def get_vector(text: str) -> List[float]:
    """
    Generates a sentence embedding vector using the loaded local model.
    Falls back to a simple placeholder if the model failed to load.
    """
    if model:
        # Use the real model if it's loaded
        return model.encode(text).tolist()
    else:
        # Placeholder logic for when the real model is missing
        vec = [0.0] * 32
        text_norm = text.lower().strip()
        for i, char in enumerate(text_norm[:32]):
            vec[i] = ord(char) / 128.0
        norm = math.sqrt(sum(x*x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

# ==============================================================================
# DATA LOADING AND HIERARCHICAL PREPROCESSING
# ==============================================================================

def get_level_rank(level_str: str) -> int:
    """
    Converts a heading level string ('H1', 'H2', 'body text') to a numeric
    rank for easy comparison. Lower numbers are higher importance.
    """
    if level_str and level_str.startswith('H') and level_str[1:].isdigit():
        return int(level_str[1:])
    else:
        return 100 # Treat 'body text' and others as lowest priority

def build_document_tree(outline: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
    """
    Transforms the flat outline from the 1a JSON into a hierarchical tree structure.
    Each node in the tree will contain its content and a list of its children.
    """
    if not outline: return []
    tree, path = [], {}
    for item in outline:
        level_str, rank = item.get('level', 'body text'), get_level_rank(item.get('level', 'body text'))
        node = {"document_filename": filename, "heading": item['text'], "level": level_str, "page": item['page'], "content_text": "", "children": []}
        if rank < 100:
            parent_rank = max([r for r in path if r < rank] or [0])
            if parent_rank > 0: path[parent_rank]['children'].append(node)
            else: tree.append(node)
            path[rank] = node
            for r in list(path.keys()):
                if r > rank: del path[r]
        else:
            parent_rank = max(path.keys()) if path else 0
            if parent_rank > 0:
                path[parent_rank]['content_text'] += f"\n{item['text']}"
                path[parent_rank]['content_text'] = path[parent_rank]['content_text'].strip()
    return tree

def precompute_vectors_and_flatten(nodes: List[Dict], path_context: str, flat_list: List[Dict]):
    """
    Traverses the tree ONCE to pre-compute all contextual vectors and create a flat list.
    This is a major performance optimization.
    """
    for node in nodes:
        # Create context with the full path of headings for better vector representation
        contextual_text = f"{path_context} -> {node['heading']}\n{node['content_text']}"
        node['contextual_vector'] = get_vector(contextual_text)
        
        # Add a copy of the node (without children) to the flat list
        node_copy = node.copy()
        node_copy.pop('children', None)
        flat_list.append(node_copy)
        
        # Recurse into children
        if node['children']:
            new_path_context = f"{path_context} -> {node['heading']}"
            precompute_vectors_and_flatten(node['children'], new_path_context, flat_list)

def load_1a_output(filepath: str) -> Dict[str, Any]:
    """Loads and parses the JSON output from the 1a script."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found."); return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file."); return None

# ==============================================================================
# CORE LOGIC (Optimized Search, Cosine Similarity, Individual LLM Calls)
# ==============================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot_product / (norm_a * norm_b)

def generate_search_steps_with_ollama(persona: str, job: str) -> List[str]:
    """Uses the LLM to generate a list of key topics/steps."""
    ollama_api_url = "http://localhost:11434/api/generate"
    prompt = f"""**System Prompt:** You are an expert research analyst. Your task is to break down a complex job into a series of actionable research steps or key topics. **Persona:** {persona} **Job to be Done:** {job} **Instruction:** Based on the persona and their job, generate a concise list of the 5 to 7 most important, high-level topics or steps this person would need to investigate in a document to accomplish their goal. These steps should be ideal queries for a semantic search. **Output Format:** Return your answer ONLY as a valid JSON object containing a single key "steps" which holds a list of strings. Do not include any other text, explanation, or markdown formatting. **Example:** {{\"steps\": [\"Core model architecture and components\", \"Key innovations and differences from previous models\", \"Training data and dataset specifics\", \"Optimizer and regularization techniques\", \"Performance benchmarks and results\"]}} **Your Turn:**"""
    payload = {"model": "gemma3:1b", "prompt": prompt, "stream": False, "format": "json"}
    
    print("Generating search steps with Ollama...")
    try:
        response = requests.post(ollama_api_url, json=payload, timeout=60)
        response.raise_for_status()
        response_text = response.json().get("response", "{}").strip()
        response_data = json.loads(response_text)
        steps = response_data.get("steps", [])
        if not steps or not isinstance(steps, list):
            print("Warning: LLM did not return a valid list of steps."); return []
        print(f"  -> Generated {len(steps)} steps.")
        return steps
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Error during step generation: {e}"); return []

def generate_refined_text_with_ollama(persona: str, job: str, section_title: str, section_text: str) -> str:
    """Generates a refined analysis for a single text section."""
    ollama_api_url = "http://localhost:11434/api/generate"
    prompt = f"""**Persona:** {persona}\n**Task:** {job}\n**Document Section Title:** {section_title}\n**Section Content:**\n---\n{section_text}\n---\n**Instruction:** Based on the persona and task, analyze the provided document section. Refine and summarize the key information that is directly relevant to the task. Focus only on the most critical points. Do not add conversational filler."""
    payload = {"model": "gemma3:1b", "prompt": prompt, "stream": False}
    
    try:
        response = requests.post(ollama_api_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "Error: No response from Ollama.").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama for refinement: {e}")
        return f"Error: Could not connect to local Ollama server at {ollama_api_url}."

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # --- Performance Tuning Parameters ---
    TOP_P_THRESHOLD = 0.9 
    MIN_SCORE_CUTOFF = 0.7
    MAX_FINAL_SECTIONS = 15

    # --- Persona and Job Definition ---
    PERSONA_DEF = {"role": "ML Engineer implementing a new model from scratch."}
    JOB_DEF = {"task": "Identify the key architectural components and the training methodology of the Transformer model."}
    
    # --- File Paths ---
    INPUT_DIR = r"C:/Users/Rama/Desktop/IITM/Programming/AI/Adobe_Hackathon/content" #if os.path.exists("/app/input") else "."
    OUTPUT_DIR = r"C:/Users/Rama/Desktop/IITM/Programming/AI/Adobe_Hackathon/content" #if os.path.exists("/app/output") else "."
    input_filename = "file02_output.json"
    input_filepath = os.path.join(INPUT_DIR, input_filename)
    output_filename = "challenge1b_output.json"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)

    # --- Main Logic ---
    print(f"Loading 1a output from: {input_filepath}")
    doc_data = load_1a_output(input_filepath)
    
    if doc_data:
        search_steps = generate_search_steps_with_ollama(PERSONA_DEF['role'], JOB_DEF['task'])
        if not search_steps: print("Could not generate search steps. Aborting."); exit()
        
        step_vectors = {step: get_vector(step) for step in search_steps}
        document_tree = build_document_tree(doc_data['outline'], input_filepath)

        # 1. Pre-compute all vectors in a single pass
        flat_sections_with_vectors = []
        print("Pre-computing all contextual vectors...")
        precompute_vectors_and_flatten(document_tree, "", flat_sections_with_vectors)

        # 2. Find top-p matches for each step from the flattened list
        print("Finding top-p matches for each step...")
        all_matches = {}
        for step, step_vec in step_vectors.items():
            scored_sections = []
            for section in flat_sections_with_vectors:
                score = cosine_similarity(section['contextual_vector'], step_vec)
                if score >= MIN_SCORE_CUTOFF:
                    section_copy = section.copy()
                    section_copy['score'] = score
                    section_copy['matched_to_step'] = step
                    scored_sections.append(section_copy)
            
            scored_sections.sort(key=lambda x: x['score'], reverse=True)
            total_score = sum(s['score'] for s in scored_sections)
            if total_score > 0:
                cumulative_p = 0.0
                for s in scored_sections:
                    if cumulative_p < TOP_P_THRESHOLD:
                        cumulative_p += s['score'] / total_score
                        key = s['heading']
                        if key not in all_matches or all_matches[key]['score'] < s['score']:
                            all_matches[key] = s
                    else: break
        
        # 3. Consolidate, rank, and limit the final list
        relevant_sections = sorted(list(all_matches.values()), key=lambda x: x['score'], reverse=True)
        final_sections_to_process = relevant_sections[:MAX_FINAL_SECTIONS]

        # 4. Build the final JSON output
        output_json = {
            "metadata": {
                "input_documents": [input_filepath],
                "persona": PERSONA_DEF['role'],
                "job_to_be_done": JOB_DEF['task'],
                "generated_search_steps": search_steps,
                "processing_timestamp": datetime.datetime.now().isoformat()
            },
            "extracted_section": [],
            "sub-section_analysis": [] 
        }

        print(f"Found {len(final_sections_to_process)} sections to process. Now generating individual summaries...")
        for i, section in enumerate(final_sections_to_process):
            rank = i + 1
            output_json["extracted_section"].append({
                "document": section['document_filename'],
                "page_number": section['page'],
                "section_title": section['heading'],
                "best_matching_step": section['matched_to_step'],
                "relevance_score": section['score'],
                "importance_rank": rank
            })
            
            # 5. *** CALLING LLM FOR EACH SECTION INDIVIDUALLY ***
            print(f"  [{rank}/{len(final_sections_to_process)}] Analyzing '{section['heading']}'...")
            refined_text = generate_refined_text_with_ollama(
                PERSONA_DEF['role'], JOB_DEF['task'], section['heading'], section['content_text']
            )
            
            output_json["sub-section_analysis"].append({
                "document": section['document_filename'],
                "page_number": section['page'],
                "refined_text": refined_text
            })

        # 6. Save Output to File
        print(f"\nSaving final output to: {output_filepath}")
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, indent=2)
            print("Successfully saved output.")
        except IOError as e:
            print(f"Error saving output file: {e}")
