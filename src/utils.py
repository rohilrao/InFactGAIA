import os
import json
from pathlib import Path
import json
import logging
from pathlib import Path
from AnthropicInFactNode import AnthropicInFactNode
from GptInFactNode import GptInFactNode
from DeepSeekInFactNode import DeepSeekInFactNode
from InFactRenderer import InFactRenderer
from IPython.display import HTML, display
from openai import OpenAI

from typing import Any, Dict, Union, List, Tuple
import json
import logging
from pathlib import Path
import os
import base64
import pandas as pd
import numpy as np
from datetime import datetime
from autogen.code_utils import extract_code
import numpy as np
from dataclasses import dataclass
import math
from typing import List, Tuple, Dict, Optional
import json
import pdfplumber

import os
import json
import base64
import openai
import pandas as pd
from pathlib import Path
from typing import Dict
from anthropic import Anthropic


def parse_data(file_path: str, hypothesis: str, provider: str, model: str, api_key: str) -> Dict:
    file_type = Path(file_path).suffix.lower()
    try:
        if file_type == ".csv":
            df = pd.read_csv(file_path)
            content = df.to_string()
            message_content = [{"type": "text", "text": content}]
        elif file_type in [".pdf", ".PDF"]:
            with open(file_path, "rb") as f:
                pdf_data = base64.b64encode(f.read()).decode("utf-8")
            message_content = [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                }
            ]
        elif file_type in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            with open(file_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            media_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }[file_type]
            message_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data
                    }
                }
            ]
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            message_content = [{"type": "text", "text": content}]

        prompt = f"""Extract relevant data points for evaluating the hypothesis:
        "{hypothesis}"

        Provide your response as a JSON code block, like this:
        ```json
        {{
            "numerical_values": [],
            "metadata": {{}},
            "issues": [],
            "confidence_assessment": {{
                "confidence_score": 0.75,
                "explanation": "Detailed explanation of confidence level",
                "key_strengths": [
                    "Strength 1",
                    "Strength 2"
                ],
                "key_limitations": [
                    "Limitation 1",
                    "Limitation 2"
                ]
            }}
        }}
        ```
        
        The confidence_assessment should:
        1. Include a confidence_score between 0 and 1
        2. Provide a detailed explanation of the confidence level
        3. List key strengths of the evidence
        4. List key limitations or potential issues
        
        The overall JSON should include:
        1. Extracted numerical values and their uncertainties
        2. Relevant metadata (source quality, methodology, etc.)
        3. Any potential issues or biases in the data
        """

        message_content.append({"type": "text", "text": prompt})

        try:
            if provider == "GPT":
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model=model,
                    max_tokens=8192,
                    temperature=0.1,
                    messages=[{"role": "user", "content": message_content}],
                )
                response_text = response["choices"][0]["message"]["content"]
            elif provider == "Anthropic":
                client = Anthropic(api_key=api_key)
                message = client.messages.create(
                    model=model,
                    max_tokens=8192,
                    temperature=0.1,
                    messages=[{"role": "user", "content": message_content}],
                )
                response_text = message.content[0].text
            else:
                return {"error": "Unsupported LLM provider"}
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "rate_limit_error" in error_message:
                for item in message_content:
                    if item["type"] == "text":
                        c = item["text"]
                        if len(c) > 10000:
                            item["text"] = c[:10000] + "\n... [truncated] ..."
                try:
                    if provider == "GPT":
                        response = openai.ChatCompletion.create(
                            model=model,
                            max_tokens=8192,
                            temperature=0.1,
                            messages=[{"role": "user", "content": message_content}],
                        )
                        response_text = response["choices"][0]["message"]["content"]
                    elif provider == "Anthropic":
                        client = Anthropic(api_key=api_key)
                        message = client.messages.create(
                            model=model,
                            max_tokens=8192,
                            temperature=0.1,
                            messages=[{"role": "user", "content": message_content}],
                        )
                        response_text = message.content[0].text
                    else:
                        return {"error": "Unsupported LLM provider"}
                except Exception as retry_error:
                    return {
                        "error": "Rate limit exceeded even after truncation",
                        "details": str(retry_error),
                    }
            else:
                return {"error": error_message}

        if not response_text:
            return {"error": "No response from LLM"}

        try:
            extracted_blocks = extract_code(response_text)
            json_str = None
            for lang, block in extracted_blocks:
                if lang.lower() in ["json", ""]:
                    try:
                        json.loads(block)
                        json_str = block
                        break
                    except json.JSONDecodeError:
                        continue
            if not json_str:
                json_str = response_text
            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError:
            return {
                "extraction_error": "Failed to parse LLM response",
                "raw_response": response_text
            }
    except Exception as e:
        return {"error": str(e)}
    

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
                    processed_files = {p.replace("My Drive", "MyDrive") for p in data}  # Fix Google Drive path issue
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
        elif node_type == "deepseek":
            node = DeepSeekInFactNode.load(str(node_state_path), api_key=api_key, model = model)
    else:
        logging.info(f"✨ Creating new {node_type} InFact node...")
        print(f"✨ Creating new {node_type} InFact node...")
        if node_type == "anthropic":
            node = AnthropicInFactNode(hypothesis=hypothesis, api_key=api_key,model=model, log_level=log_level)
        elif node_type == "gpt":
            node = GptInFactNode(hypothesis=hypothesis, api_key=api_key, model=model, log_level=log_level)
        elif node_type == "deepseek":
            node = DeepSeekInFactNode(hypothesis=hypothesis, api_key=api_key, model=model, log_level=log_level)

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

def get_latest_html_path(node_type, results_dir):
    """Finds the most recent HTML result file and returns its file path."""
    
    node_dir = results_dir / node_type  # Node-specific results directory
    
    # Get all HTML files in the results directory
    html_files = list(node_dir.glob("*.html"))  
    
    if not html_files:
        print(f"🚫 No HTML result files found for {node_type}.")
        return None  # ✅ Return None if no files found

    # Get the most recently created HTML file
    latest_html_file = max(html_files, key=os.path.getctime)

    print(f"✅ Latest HTML result file path: {latest_html_file}")  # Debugging statement

    return str(latest_html_file)  # ✅ Return file path instead of rendering it        


'''
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
    results_dir = hypo_path / "results" 
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
    all_evidence_files = {p.replace("My Drive", "MyDrive") for p in all_evidence_files} # Fix Google Drive path issue

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
''' 


