import json
import base64
import logging
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import pandas as pd
from bson.objectid import ObjectId

# Import your existing call_llm function
from infact_utils import call_llm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FileProcessor")

def extract_code(text: str) -> List[Tuple[str, str]]:
    """Extract code blocks from text using markdown-style code block syntax."""
    import re
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def process_file(db, fs, file_id, api_key, provider="openai", model="gpt-4o"):
    """
    Process a file from GridFS, extract metadata, and use LLM to parse its contents.
    
    Args:
        db: MongoDB database connection
        fs: GridFS connection
        file_id: ID of the file in GridFS
        api_key: API key for LLM provider
        provider: LLM provider name (default: "openai")
        model: LLM model to use
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting to process file with ID: {file_id}")
    
    try:
        # Status should already be set to "processing" from the UI
        # Just verify it's in processing state
        file_status = file_metadata.get("metadata", {}).get("status", "")
        if file_status != "processing":
            logger.info(f"File status is {file_status}, updating to 'processing'")
            db.fs.files.update_one(
                {"_id": file_id},
                {"$set": {"metadata.status": "processing"}}
            )
        
        # Get file from GridFS
        grid_file = fs.get(file_id)
        filename = grid_file.filename
        file_content = grid_file.read()
        
        # Get hypothesis from file metadata
        file_metadata = db.fs.files.find_one({"_id": file_id})
        if not file_metadata or "metadata" not in file_metadata:
            raise ValueError(f"File metadata not found for file ID: {file_id}")
            
        hypothesis = file_metadata["metadata"].get("hypothesis_text", "")
        if not hypothesis:
            raise ValueError("Hypothesis text not found in file metadata")
        
        logger.info(f"Processing file: {filename} for hypothesis: {hypothesis[:50]}...")
        
        # Create a temporary file to work with
        file_ext = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Extract basic metadata
            logger.info("Extracting metadata...")
            metadata = _extract_metadata(temp_file_path)
            
            # Parse file data using LLM
            logger.info("Parsing file data using LLM...")
            parsed_data = _parse_data(temp_file_path, hypothesis, api_key, provider, model)
            
            # Update file in GridFS with the results - proper status transition to ready_for_analysis
            db.fs.files.update_one(
                {"_id": file_id},
                {"$set": {
                    "metadata.status": "ready_for_analysis",
                    "metadata.parsed_data": parsed_data,
                    "metadata.extracted_metadata": metadata,
                    "metadata.processing_date": datetime.now().isoformat(),
                    "metadata.parsing_complete": True,
                    "metadata.processing_provider": provider,
                    "metadata.processing_model": model
                }}
            )
            
            logger.info(f"Updated file status to 'ready_for_analysis'")
            
            logger.info(f"File processing completed successfully for {filename}")
            return {
                "success": True,
                "file_id": str(file_id),
                "filename": filename,
                "metadata": metadata,
                "parsed_data": parsed_data
            }
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        
        # Update file status to error
        db.fs.files.update_one(
            {"_id": file_id},
            {"$set": {
                "metadata.status": "error",
                "metadata.error_message": str(e),
                "metadata.parsing_complete": False
            }}
        )
        
        return {
            "success": False,
            "file_id": str(file_id) if file_id else None,
            "error": str(e)
        }

def _parse_data(data_file: str, hypothesis: str, api_key: str, provider: str, model: str) -> Dict:
    """Parse different file types using LLM assistance."""
    logger.info(f"Parsing data file: {data_file}")
    file_type = Path(data_file).suffix.lower()

    try:
        # Prepare file content based on file type
        if file_type == '.csv':
            logger.debug("Processing CSV file")
            df = pd.read_csv(data_file)
            content = df.to_string()
            
        elif file_type in ['.pdf', '.PDF']:
            logger.debug("Processing PDF file")
            with open(data_file, 'rb') as f:
                pdf_data = base64.b64encode(f.read()).decode('utf-8')
            # Since we're using text-only APIs, describe PDF content
            content = f"[PDF file: {Path(data_file).name}, size: {os.path.getsize(data_file)} bytes]"
            
        elif file_type in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            logger.debug(f"Processing image file of type {file_type}")
            # Since we're using text-only APIs, describe image content
            content = f"[Image file: {Path(data_file).name}, type: {file_type}, size: {os.path.getsize(data_file)} bytes]"
            
        else:
            logger.debug(f"Processing text file of type {file_type}")
            with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

        # Create analysis prompt
        prompt = f"""
        Extract relevant data points for evaluating the hypothesis:
        "{hypothesis}"

        Here is the content to analyze:
        {content[:10000]}  # Limiting content length to avoid token limits

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

        logger.debug(f"Prepared prompt for LLM")
        logger.info(f"Calling {provider} LLM with model {model}")
        
        # Call LLM using your existing function
        response_text = call_llm(provider, api_key, model, prompt)
        logger.debug(f"Received API response of length: {len(response_text)}")

        # Try to extract JSON using the extract_code function
        extracted_blocks = extract_code(response_text)

        # Look for JSON blocks
        json_str = None
        for lang, block in extracted_blocks:
            if lang.lower() in ['json', '']:
                try:
                    # Try to parse as JSON to validate
                    parsed = json.loads(block)
                    json_str = block
                    break
                except json.JSONDecodeError:
                    continue

        # If no valid JSON block found, try parsing the whole response
        if not json_str:
            logger.warning("No JSON code block found, trying to parse entire response")
            try:
                # Find anything that looks like JSON by searching for opening/closing braces
                import re
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    potential_json = json_match.group(1)
                    parsed = json.loads(potential_json)
                    json_str = potential_json
                else:
                    raise ValueError("No JSON-like structure found in response")
            except Exception:
                logger.error("Failed to parse response as JSON")
                return {
                    "extraction_error": "Failed to parse LLM response",
                    "raw_response": response_text[:1000]  # Include truncated response
                }

        parsed_data = json.loads(json_str)
        logger.debug(f"Successfully parsed JSON data")
        return parsed_data

    except Exception as e:
        logger.error(f"Error in _parse_data: {str(e)}", exc_info=True)
        return {
            "extraction_error": str(e),
            "file_type": file_type
        }

def _extract_metadata(data_file: str) -> Dict[str, Any]:
    """Extract metadata from the data file."""
    logger.info(f"Extracting metadata from {data_file}")

    try:
        file_path = Path(data_file)
        basic_metadata = {
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size,
            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "source_path": str(file_path.absolute())
        }

        if file_path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    if hasattr(pdf, 'metadata') and pdf.metadata:
                        basic_metadata.update({
                            "title": pdf.metadata.get('/Title', ''),
                            "author": pdf.metadata.get('/Author', ''),
                            "creator": pdf.metadata.get('/Creator', ''),
                            "producer": pdf.metadata.get('/Producer', ''),
                            "creation_date": pdf.metadata.get('/CreationDate', ''),
                            "modification_date": pdf.metadata.get('/ModDate', ''),
                            "page_count": len(pdf.pages)
                        })
            except ImportError:
                logger.warning("PyPDF2 not installed, skipping PDF metadata extraction")
            except Exception as e:
                logger.warning(f"Error extracting PDF metadata: {str(e)}")

        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    basic_metadata.update({
                        "image_format": img.format,
                        "image_size": img.size,
                        "image_mode": img.mode
                    })
                    # Convert image info to serializable format
                    img_info = {}
                    for k, v in img.info.items():
                        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                            img_info[k] = v
                    basic_metadata["image_info"] = img_info
            except ImportError:
                logger.warning("Pillow not installed, skipping image metadata extraction")
            except Exception as e:
                logger.warning(f"Error extracting image metadata: {str(e)}")

        elif file_path.suffix.lower() == '.html':
            try:
                from bs4 import BeautifulSoup
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    meta_tags = {}
                    for meta in soup.find_all('meta'):
                        name = meta.get('name', meta.get('property', ''))
                        content = meta.get('content', '')
                        if name and content:
                            meta_tags[name] = content

                    title_tag = soup.title
                    title_text = title_tag.string if title_tag else ''
                    
                    basic_metadata.update({
                        "title": title_text,
                        "meta_tags": meta_tags,
                        "has_article": bool(soup.find('article')),
                        "has_main": bool(soup.find('main')),
                        "num_headers": len(soup.find_all(['h1', 'h2', 'h3'])),
                        "has_tables": bool(soup.find_all('table'))
                    })
            except ImportError:
                logger.warning("BeautifulSoup4 not installed, skipping HTML metadata extraction")
            except Exception as e:
                logger.warning(f"Error extracting HTML metadata: {str(e)}")

        elif file_path.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(file_path)
                basic_metadata.update({
                    "num_rows": len(df),
                    "num_columns": len(df.columns),
                    "column_names": list(df.columns),
                    "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "has_nulls": bool(df.isnull().any().any())
                })
            except Exception as e:
                logger.warning(f"Error extracting CSV metadata: {str(e)}")

        logger.debug(f"Extracted metadata successfully")
        return basic_metadata

    except Exception as e:
        logger.error(f"Error in _extract_metadata: {str(e)}", exc_info=True)
        return {
            "filename": Path(data_file).name,
            "error": str(e)
        }