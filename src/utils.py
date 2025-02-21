import os
import json
from pathlib import Path
import json
import logging
from pathlib import Path
from AnthropicInFactNode import AnthropicInFactNode
from GptInFactNode import GptInFactNode
from InFactRenderer import InFactRenderer
from IPython.display import HTML, display

# ✅ Function to get processed files JSON path
def get_processed_log_path(node_type, results_dir):
    node_dir = results_dir / node_type  # Node-specific results directory
    return node_dir / f"processed_files_{node_type}.json"  # ✅ Store in the correct folder

def load_processed_files(node_type, results_dir):
    """Loads the processed files list, or logs that no existing records were found."""
    
    processed_log_path = get_processed_log_path(node_type, results_dir)

    if processed_log_path.exists():
        try:
            with open(processed_log_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    processed_files = set(data)
                    logging.info(f"📂 Processed files read from {processed_log_path}: {len(processed_files)} files.")
                    print(f"📂 Processed files read from {processed_log_path}: {len(processed_files)} files.")
                    return processed_files  # ✅ Ensure correct format
                else:
                    print("⚠️ Unexpected format in processed files log. Resetting data.")
                    return set()
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Error reading processed files log: {e}. Resetting data.")
            return set()
    
    # ✅ Log when no existing processed files are found
    print(f"🚫 No existing processed files found for {node_type} at {processed_log_path}. Starting fresh.")
    return set()


def save_processed_files(processed_files, node_type, results_dir):
    processed_log_path = get_processed_log_path(node_type, results_dir)

    # ✅ Load existing processed files first
    existing_files = load_processed_files(node_type, results_dir)

    # ✅ Merge both sets
    updated_files = existing_files | processed_files  # Union of both sets

    # ✅ Save back to file
    with open(processed_log_path, "w") as f:
        json.dump(list(updated_files), f)


# ✅ Function to create or load InFact nodes dynamically
def load_or_create_node(node_type, results_dir, hypothesis, model, api_key):
    """Load or create an InFact node for GPT, Anthropic, or DeepSeek."""
    log_level = logging.DEBUG

    # ✅ Define node-specific directories
    node_dir = results_dir / node_type
    node_dir.mkdir(parents=True, exist_ok=True)
    node_state_path = node_dir / "infact_node_state.json"

    # ✅ Load existing node state
    if node_state_path.exists():
        logging.info(f"🔄 Loading existing {node_type} node state...")
        print(f"🔄 Loading existing {node_type} node state...")
        if node_type == "anthropic":
            node = AnthropicInFactNode.load(str(node_state_path), api_key=api_key, model = model)
        elif node_type == "gpt":
            node = GptInFactNode.load(str(node_state_path), api_key=api_key, model = model)
    else:
        logging.info(f"✨ Creating new {node_type} InFact node...")
        print(f"✨ Creating new {node_type} InFact node...")
        if node_type == "anthropic":
            node = AnthropicInFactNode(hypothesis=hypothesis, api_key=api_key,model=model, log_level=log_level)
        elif node_type == "gpt":
            node = GptInFactNode(hypothesis=hypothesis, api_key=api_key, model=model, log_level=log_level)

    return node, node_state_path, node_dir

# ✅ Function to display the most recent HTML result
def display_latest_html_result(node_type, results_dir):
    """Finds the most recent HTML result file and renders it inline if possible."""
    
    node_dir = results_dir / node_type  # Node-specific results directory
    
    # Get all HTML files in the results directory
    html_files = list(node_dir.glob("*.html"))  
    
    if not html_files:
        print(f"🚫 No HTML result files found for {node_type}.")
        return None

    # Get the most recently created HTML file
    latest_html_file = max(html_files, key=os.path.getctime)

    print(f"Displaying most recent HTML result for {node_type}: {latest_html_file}")

    # ✅ Try to read and render the HTML inline
    try:
        with open(latest_html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        display(HTML(html_content))  # Render inline in Colab
    except Exception as e:
        print(f"❌ Error displaying HTML: {e}")

# ✅ Function to process new evidence files and display results
def process_evidence(node_type, hypothesis_folder_name, base_dir, api_key, model, hypothesis):
    """
    Manually trigger processing of new evidence files for a specific node type.

    Parameters:
    - node_type (str): "gpt", "anthropic", or "deepseek"
    - hypothesis_folder_name (str): Folder name for the hypothesis
    - base_dir (Path): The base directory where hypothesis folders are stored
    - api_key (str): API key for the selected model
    - hypothesis (str): The hypothesis text
    """

    # ✅ Dynamically construct paths
    hypo_path = base_dir / hypothesis_folder_name
    results_dir = hypo_path / "results" / node_type
    evidence_dir = hypo_path / "evidence"

    print(f"📂 Hypothesis path: {hypo_path}")
    print(f"\n📂 Results directory: {results_dir}")
    print(f"\n📂 Evidence directory: {evidence_dir}")

    # ✅ Ensure directories exist
    results_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    node, node_state_path, node_dir = load_or_create_node(node_type, results_dir, hypothesis, model, api_key)
    processed_files = load_processed_files(node_type, results_dir)

    # Define supported file types
    supported_file_types = {'.csv', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.html', '.htm'}

    # Get all supported files
    all_evidence_files = {str(p) for p in evidence_dir.glob("*.*") if p.suffix.lower() in supported_file_types}

    if not all_evidence_files:  # ✅ Handle case when no files exist
        logging.info(f"🚫 No evidence files found in {evidence_dir}. Nothing to process.")
        print(f"🚫 No evidence files found in {evidence_dir}. Nothing to process.")
        return

    new_files = all_evidence_files - processed_files  # Only unprocessed files

    if not new_files:
        logging.info(f"✅ No new evidence files found for {node_type}.")
        print(f"✅ No new evidence files found for {node_type}.")
        return

    logging.info(f"📂 Found {len(new_files)} new evidence files for {node_type}.")
    print(f"📂 Found {len(new_files)} new evidence files for {node_type}.")

    for evidence_file in new_files:
        logging.info(f"🔄 Processing file: {evidence_file} with {node_type}...")
        print(f"🔄 Processing file: {evidence_file} with {node_type}...")

        node.process_data(evidence_file)  # Process new data

        # Save updated node state
        node.save(str(node_state_path))

        # Save analysis results
        # analysis_path = node_dir / f"analysis_{Path(evidence_file).stem}.json"
        # node.save(str(analysis_path))

        # Render output
        renderer = InFactRenderer()
        output_file = node_dir / f"analysis_{Path(evidence_file).stem}.html"
        renderer.render_analysis(node, str(output_file))

        # Mark as processed
        processed_files.add(evidence_file)

    save_processed_files(processed_files, node_type, results_dir)
    logging.info(f"✅ Processing completed for {node_type}.")
    print(f"✅ Processing completed for {node_type}.")