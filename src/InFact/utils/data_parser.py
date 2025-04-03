import json
import base64
import pandas as pd
from io import BytesIO, StringIO
from bson.objectid import ObjectId
from typing import Dict, Any, Union, List
import logging
from autogen.code_utils import extract_code
import gridfs


def parse_data_from_db(db, file_id: Union[str, ObjectId], hypothesis: str, llm_provider, logger) -> Dict:
    """
    Parse different file types using LLM assistance, retrieving file content directly from the database.
    
    Args:
        db: MongoDB database connection
        file_id: ObjectId or string ID of the file in GridFS
        hypothesis: The hypothesis being evaluated
        llm_provider: LLM provider instance
        logger: Logger instance
        
    Returns:
        Dict: Parsed data
    """
    logger.info(f"Parsing data file from DB with ID: {file_id}")
    
    # Ensure file_id is an ObjectId
    if isinstance(file_id, str):
        file_id = ObjectId(file_id)
    
    try:
        # Get file information from DB
        file_doc = db.fs.files.find_one({"_id": file_id})
        if not file_doc:
            logger.error(f"File with ID {file_id} not found in database")
            raise FileNotFoundError(f"File with ID {file_id} not found in database")
        else:
            logger.info(f"File with ID {file_id} found successfully")
        
        filename = file_doc["filename"]
        file_type = "." + filename.split(".")[-1].lower() if "." in filename else ""
        
        # Create a GridFS instance to retrieve the file
        fs = gridfs.GridFS(db)
        
        # Retrieve file content from GridFS
        print(f"Retrieving content for file '{filename}' (type: {file_type})")
        if not fs.exists(file_id):
            print(f"ERROR: File with ID {file_id} exists in metadata but not in GridFS")
            raise FileNotFoundError(f"File content for ID {file_id} not found in GridFS")
            
        # Use proper GridFS access
        with fs.get(file_id) as grid_out:
            file_content = grid_out.read()
        
        # Prepare content based on file type
        message_content = _prepare_db_file_content(file_content, filename, file_type, logger)
        
        # Add analysis prompt
        prompt = f"""
        Extract relevant data points for evaluating the hypothesis:
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

        # Add prompt to message content if it's a list of structured content
        if isinstance(message_content, list):
            message_content.append({"type": "text", "text": prompt})
        else:
            # If it's just text, append the prompt
            message_content += "\n\n" + prompt
        

        print("THIS CODE IS RUNNING")
        print(message_content)
        print(len(message_content)
        print("THIS CODE IS RUNNING1"))
        # Log the prepared prompt
        logger.debug(f"Prepared prompt for parsing: {prompt}")
        logger.debug(f"Prepared message content for parsing: {message_content}")
        # Log the file content length
        logger.debug(f"File content length: {len(file_content)} bytes")
        # Log the file type

        logger.debug(f"Prepared prompt for parsing")

        # Send to LLM with retry logic
        response_text = llm_provider.send_with_retry(message_content)
        logger.debug(f"Received API response for parsing")

        # Extract JSON from response
        parsed_data = _extract_json_from_response(response_text, logger)
        logger.debug(f"Successfully parsed JSON data")
        
        return parsed_data

    except Exception as e:
        logger.error(f"Error in parse_data_from_db: {str(e)}", exc_info=True)
        raise


def _prepare_db_file_content(file_content: bytes, filename: str, file_type: str, logger) -> Any:
    """
    Prepare file content based on file type, working directly with binary content from DB.
    
    Args:
        file_content: Binary content of the file from GridFS
        filename: Name of the file
        file_type: File extension (e.g., '.csv')
        logger: Logger instance
        
    Returns:
        Content in the format expected by the LLM provider
    """
    def truncate_content(content: str, max_length: int = 15000) -> str:
        """Truncate content to a maximum length."""
        if len(content) > max_length:
            logger.warning(f"Content length ({len(content)}) exceeds max length ({max_length}), truncating.")
            return content[:max_length] + "\n... [truncated] ..."
        return content

    try:
        if file_type == '.csv':
            logger.debug("Processing CSV file")
            csv_stream = StringIO(file_content.decode('utf-8', errors='replace'))
            df = pd.read_csv(csv_stream)
            content = df.to_string()
            return [{"type": "text", "text": truncate_content(content)}]

        elif file_type in ['.pdf', '.PDF']:
            logger.debug("Processing PDF file")
            # Try to determine if provider supports direct PDF handling
            try:
                pdf_data = base64.b64encode(file_content).decode('utf-8')
                return [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data
                        }
                    }
                ]
            except Exception as e:
                logger.warning(f"Direct PDF handling failed, falling back to text extraction: {str(e)}")
                
                # Fall back to text extraction
                extracted_text = _extract_text_from_pdf_bytes(file_content, logger)
                return [{"type": "text", "text": truncate_content(extracted_text)}]

        elif file_type in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            logger.debug(f"Processing image file of type {file_type}")
            img_data = base64.b64encode(file_content).decode('utf-8')
            media_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }[file_type]
            return [
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
            logger.debug(f"Processing text file of type {file_type}")
            try:
                content = file_content.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                content = f"Unable to decode file content as UTF-8. File may be binary: {filename}"
            
            return [{"type": "text", "text": truncate_content(content)}]

    except Exception as e:
        logger.error(f"Error preparing file content: {str(e)}", exc_info=True)
        raise


def _extract_text_from_pdf_bytes(pdf_content: bytes, logger) -> str:
    """
    Extract text from PDF using multiple methods with fallbacks, working directly with PDF bytes.
    
    Args:
        pdf_content: Binary content of the PDF
        logger: Logger instance
        
    Returns:
        str: Extracted text
    """
    logger.debug("Extracting text from PDF bytes")
    
    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF
        logger.debug("Using PyMuPDF for text extraction")
        with BytesIO(pdf_content) as pdf_stream:
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {str(e)}, trying pdfplumber")
    
    # Try pdfplumber next
    try:
        import pdfplumber
        logger.debug("Using pdfplumber for text extraction")
        with BytesIO(pdf_content) as pdf_stream:
            with pdfplumber.open(pdf_stream) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {str(e)}, trying textract")
    
    # Try textract as last resort - requires saving to temp file since textract needs a file path
    try:
        import tempfile
        import os
        import textract
        
        logger.debug("Using textract for text extraction (with temp file)")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_content)
        
        try:
            # Extract text and clean up
            text = textract.process(temp_path, method='pdfminer').decode('utf-8')
            return text
        finally:
            # Ensure temp file is deleted even if extraction fails
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp PDF file: {str(e)}")
    except Exception as e:
        logger.error(f"All PDF extraction methods failed: {str(e)}")
        return "Error: Unable to extract text from PDF using available methods."


def _extract_json_from_response(response_text: str, logger) -> Dict:
    """
    Extract JSON data from LLM response.
    
    Args:
        response_text: Text response from LLM
        logger: Logger instance
        
    Returns:
        Dict: Parsed JSON data
    """
    try:
        # Try to extract JSON using autogen
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
                parsed = json.loads(response_text)
                json_str = response_text
            except json.JSONDecodeError:
                # Try to find JSON in response using regex
                import re
                logger.warning("Failed to parse response as JSON, trying regex extraction")
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, response_text)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                        json_str = match.group(1)
                    except json.JSONDecodeError:
                        logger.error("Regex extraction failed to find valid JSON")
                        return {
                            "extraction_error": "Failed to parse LLM response",
                            "raw_response": response_text
                        }
                else:
                    logger.error("Failed to parse response as JSON")
                    return {
                        "extraction_error": "Failed to parse LLM response",
                        "raw_response": response_text
                    }

        parsed_data = json.loads(json_str)
        return parsed_data

    except Exception as e:
        logger.error(f"Error extracting JSON from response: {str(e)}", exc_info=True)
        return {
            "extraction_error": f"Error: {str(e)}",
            "raw_response": response_text
        }


# Standalone version for direct use without InFactNode
def parse_db_standalone(db, file_id: Union[str, ObjectId], hypothesis: str, provider: str, model: str, api_key: str) -> Dict:
    """
    Parse file data directly from the database and extract information relevant to a hypothesis.
    Standalone version for use without requiring an InFactNode instance.
    
    Args:
        db: MongoDB database connection
        file_id: ObjectId or string ID of the file in GridFS
        hypothesis: The hypothesis to evaluate against
        provider: "GPT" or "Anthropic" or "DeepSeek"
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet")
        api_key: API key for the provider
        
    Returns:
        Dict containing parsed data or error information
    """
    # Setup simple logger
    logger = logging.getLogger("parse_db_standalone")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Ensure file_id is ObjectId
    if isinstance(file_id, str):
        file_id = ObjectId(file_id)
    
    file_doc = db.fs.files.find_one({"_id": file_id})
    if not file_doc:
        return {"error": f"File with ID {file_id} not found in database"}
    
    logger.info(f"Parsing file: {file_doc['filename']} (ID: {file_id}) using {provider}/{model}")
    
    try:
        # Import providers dynamically to avoid circular imports
        if provider.lower() == "gpt" or provider.lower() == "openai":
            from ..providers.openai_provider import OpenAIProvider
            llm_provider = OpenAIProvider(api_key=api_key, model=model)
        elif provider.lower() == "anthropic":
            from ..providers.anthropic_provider import AnthropicProvider
            llm_provider = AnthropicProvider(api_key=api_key, model=model)
        else:
            return {"error": f"Unsupported provider: {provider}"}
            
        # Use the core parse_data_from_db function with the appropriate provider
        return parse_data_from_db(db, file_id, hypothesis, llm_provider, logger)
        
    except Exception as e:
        logger.error(f"Error in parse_db_standalone: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}