import os
import json
from pathlib import Path
import json
import logging
from pathlib import Path
from AnthropicInFactNode import AnthropicInFactNode
from GptInFactNode import GptInFactNode
from DeepSeekInFactNode import DeepSeekInFactNode
from InFact.utils.InFactRenderer import InFactRenderer
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
    """
    Parse file data and extract information relevant to a hypothesis using the specified LLM.
    
    Args:
        file_path: Path to the file to be analyzed
        hypothesis: The hypothesis to evaluate against
        provider: "GPT" or "Anthropic"
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet")
        api_key: API key for the provider
        
    Returns:
        Dict containing parsed data or error information
    """
    print(f"DEBUG - Parsing file: {file_path} using {provider}/{model}")
    file_type = Path(file_path).suffix.lower()
    
    try:
        # Prepare the base prompt text
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

        # Handle different providers and file types appropriately
        if provider == "GPT":
            return _parse_with_openai(file_path, file_type, hypothesis, model, api_key, prompt)
        elif provider == "Anthropic":
            return _parse_with_anthropic(file_path, file_type, hypothesis, model, api_key, prompt)
        else:
            return {"error": f"Unsupported provider: {provider}"}
            
    except Exception as e:
        print(f"DEBUG - Error in parse_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def _parse_with_openai(file_path, file_type, hypothesis, model, api_key, prompt):
    """Handle parsing with OpenAI models"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    
    try:
        # Handle different file types for OpenAI
        if file_type == ".csv":
            try:
                # Read CSV as text
                df = pd.read_csv(file_path)
                content = df.to_string()
                
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": content},
                    {"type": "text", "text": prompt}
                ]}]
            except Exception as e:
                print(f"DEBUG - Error reading CSV, falling back to basic text: {str(e)}")
                # Fall back to simple text reading
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": f"CSV Content (read as text):\n{content[:15000]}"},
                    {"type": "text", "text": prompt}
                ]}]
            
        elif file_type in [".pdf", ".PDF"]:
            # For PDFs with OpenAI, try multiple approaches with fallbacks
            
            # First, try to determine if we can use vision capabilities
            has_vision = "gpt-4" in model and "vision" in model or "gpt-4o" in model
            
            # Define a text extraction fallback
            def extract_text_fallback():
                print("DEBUG - Using text extraction fallback for PDF")
                try:
                    # Try PyMuPDF first
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(file_path)
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        return text
                    except Exception as e:
                        print(f"DEBUG - PyMuPDF failed: {str(e)}, trying pdfplumber")
                        
                    # Fall back to pdfplumber
                    try:
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            text = ""
                            for page in pdf.pages:
                                text += page.extract_text() or ""
                        return text
                    except Exception as e:
                        print(f"DEBUG - pdfplumber failed: {str(e)}, trying textract")
                        
                    # Fall back to textract as last resort
                    try:
                        import textract
                        text = textract.process(file_path, method='pdfminer').decode('utf-8')
                        return text
                    except Exception as e:
                        print(f"DEBUG - textract failed: {str(e)}")
                        
                    # If all else fails, inform about the issue
                    return "Error: Unable to extract text from PDF using multiple methods."
                        
                except ImportError as e:
                    print(f"DEBUG - PDF extraction libraries not available: {str(e)}")
                    return "Error: Required PDF processing libraries not installed."
            
            # Start with preferred method based on model capabilities
            if has_vision:
                try:
                    # Option 1: Use the vision capability to handle PDFs
                    import fitz  # PyMuPDF
                    
                    doc = fitz.open(file_path)
                    content_parts = []
                    
                    # Only process first few pages to avoid token limits
                    max_pages = min(5, len(doc))
                    
                    for page_num in range(max_pages):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                        img_data = pix.tobytes("png")
                        
                        # Convert to base64
                        img_b64 = base64.b64encode(img_data).decode('utf-8')
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        })
                    
                    # Add text prompt after all images
                    content_parts.append({"type": "text", "text": prompt})
                    messages = [{"role": "user", "content": content_parts}]
                    
                except Exception as e:
                    print(f"DEBUG - Failed to process PDF as images: {str(e)}, falling back to text extraction")
                    # Fall back to text extraction
                    extracted_text = extract_text_fallback()
                    messages = [{"role": "user", "content": [
                        {"type": "text", "text": f"PDF Content:\n{extracted_text[:15000]}"},
                        {"type": "text", "text": prompt}
                    ]}]
            else:
                # For non-vision models, go straight to text extraction
                extracted_text = extract_text_fallback()
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": f"PDF Content:\n{extracted_text[:15000]}"},
                    {"type": "text", "text": prompt}
                ]}]
        
        elif file_type in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            # Handle images - OpenAI requires image_url format
            with open(file_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            
            media_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }[file_type]
            
            messages = [{"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{img_data}"
                    }
                },
                {"type": "text", "text": prompt}
            ]}]
            
        else:
            # Handle text-based files
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            messages = [{"role": "user", "content": [
                {"type": "text", "text": content},
                {"type": "text", "text": prompt}
            ]}]
        
        # Make the API call
        print(f"DEBUG - Calling OpenAI API with model {model}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.1,
        )
        
        response_text = response.choices[0].message.content
        return _extract_json_response(response_text)
        
    except Exception as e:
        print(f"DEBUG - OpenAI API error: {str(e)}")
        if "429" in str(e) or "rate_limit" in str(e).lower():
            # Try with truncated content
            try:
                # Simplify messages to just the prompt with minimal file content
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": f"[Limited file content due to size constraints]"},
                    {"type": "text", "text": prompt}
                ]}]
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.1,
                )
                
                response_text = response.choices[0].message.content
                return _extract_json_response(response_text)
                
            except Exception as retry_error:
                return {
                    "error": "Rate limit exceeded even after truncation",
                    "details": str(retry_error)
                }
        else:
            return {"error": str(e)}

def _parse_with_anthropic(file_path, file_type, hypothesis, model, api_key, prompt):
    """Handle parsing with Anthropic Claude models"""
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    
    # Define a helper function for PDF text extraction as fallback
    def extract_text_from_pdf():
        print("DEBUG - Using text extraction fallback for PDF with Claude")
        try:
            # Try PyMuPDF first
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
            except Exception as e:
                print(f"DEBUG - PyMuPDF failed: {str(e)}, trying pdfplumber")
                
            # Fall back to pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                return text
            except Exception as e:
                print(f"DEBUG - pdfplumber failed: {str(e)}, trying textract")
                
            # Fall back to textract as last resort
            try:
                import textract
                text = textract.process(file_path, method='pdfminer').decode('utf-8')
                return text
            except Exception as e:
                print(f"DEBUG - All PDF extraction methods failed: {str(e)}")
                return "Error: Unable to extract text from PDF."
                
        except ImportError as e:
            print(f"DEBUG - PDF extraction libraries not available: {str(e)}")
            return "Error: Required PDF processing libraries not installed."
    
    try:
        # Prepare the message content based on file type
        if file_type == ".csv":
            try:
                df = pd.read_csv(file_path)
                content = df.to_string()
                message_content = [
                    {"type": "text", "text": content},
                    {"type": "text", "text": prompt}
                ]
            except Exception as e:
                print(f"DEBUG - Error reading CSV, falling back to basic text: {str(e)}")
                # Fall back to simple text reading
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                message_content = [
                    {"type": "text", "text": f"CSV Content (read as text):\n{content[:15000]}"},
                    {"type": "text", "text": prompt}
                ]
            
        elif file_type in [".pdf", ".PDF"]:
            # First try native PDF handling for Claude
            try:
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
                    },
                    {"type": "text", "text": prompt}
                ]
                
                # Test the API call with minimal prompt to see if PDF handling works
                # This lets us catch file type errors before sending the full content
                test_response = client.messages.create(
                    model=model,
                    max_tokens=10,  # Minimal tokens for test
                    messages=[{
                        "role": "user", 
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_data
                                }
                            },
                            {"type": "text", "text": "Is this PDF readable? Just say yes or no."}
                        ]
                    }]
                )
                
                # If we get here, PDF handling works, continue with original message_content
                print("DEBUG - Claude direct PDF handling successful")
                
            except Exception as e:
                print(f"DEBUG - Claude PDF handling failed: {str(e)}, falling back to text extraction")
                # Extract text as fallback
                extracted_text = extract_text_from_pdf()
                message_content = [
                    {"type": "text", "text": f"PDF Content:\n{extracted_text[:25000]}"},
                    {"type": "text", "text": prompt}
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
                },
                {"type": "text", "text": prompt}
            ]
            
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
                
            message_content = [
                {"type": "text", "text": content},
                {"type": "text", "text": prompt}
            ]
        
        # Make the API call
        print(f"DEBUG - Calling Claude API with model {model}")
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.1,
            messages=[{"role": "user", "content": message_content}]
        )
        
        response_text = message.content[0].text
        return _extract_json_response(response_text)
        
    except Exception as e:
        print(f"DEBUG - Claude API error: {str(e)}")
        if "429" in str(e) or "rate_limit" in str(e).lower():
            # Try with truncated content
            try:
                # Simplify message to just the prompt
                message_content = [
                    {"type": "text", "text": "[Limited file content due to size constraints]"},
                    {"type": "text", "text": prompt}
                ]
                
                message = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    temperature=0.1,
                    messages=[{"role": "user", "content": message_content}]
                )
                
                response_text = message.content[0].text
                return _extract_json_response(response_text)
                
            except Exception as retry_error:
                return {
                    "error": "Rate limit exceeded even after truncation",
                    "details": str(retry_error)
                }
        else:
            return {"error": str(e)}

def _extract_json_response(response_text):
    """Extract and parse JSON from the model response"""
    import re
    import json
    
    # Look for JSON code blocks
    json_pattern = r'```(?:json)?([\s\S]*?)```'
    matches = re.findall(json_pattern, response_text)
    
    if matches:
        # Try each match until we find valid JSON
        for match in matches:
            try:
                parsed_data = json.loads(match.strip())
                return parsed_data
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON in code blocks, try to find JSON directly
    try:
        # Look for any text that might be JSON (between curly braces)
        json_candidate = re.search(r'({[\s\S]*})', response_text)
        if json_candidate:
            parsed_data = json.loads(json_candidate.group(1))
            return parsed_data
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # If all else fails, return the raw response
    return {
        "extraction_error": "Failed to parse JSON from response",
        "raw_response": response_text
    }

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


