import json
import base64
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import logging
from autogen.code_utils import extract_code
import os


def parse_data(data_file: str, hypothesis: str, llm_provider, logger) -> Dict:
    """
    Parse different file types using LLM assistance.
    
    Args:
        data_file: Path to the data file
        hypothesis: The hypothesis being evaluated
        llm_provider: LLM provider instance
        logger: Logger instance
        
    Returns:
        Dict: Parsed data or error information
    """
    logger.info(f"Parsing data file: {data_file}")
    print(f"🔍 Processing: {data_file}")
    file_type = Path(data_file).suffix.lower()

    try:
        # Prepare content based on file type
        message_content = _prepare_file_content(data_file, file_type, logger)
        
        # Check if message_content is None or empty
        if message_content is None:
            error_msg = "Failed to extract content from file"
            logger.error(error_msg)
            print(f"❌ Error: {error_msg}")
            return {"error": error_msg, "file": data_file}
            
        print(f"✅ Successfully extracted content from {data_file}")
        # Print first part of the content for debugging (limit length for readability)
        if isinstance(message_content, list) and len(message_content) > 0:
            if message_content[0].get("type") == "text":
                debug_content = message_content[0].get("text", "")[:200] + "..." if len(message_content[0].get("text", "")) > 200 else message_content[0].get("text", "")
                print(f"📄 Content preview: {debug_content}")
            else:
                print(f"📄 Content type: {message_content[0].get('type', 'unknown')}")
        else:
            debug_content = str(message_content)[:200] + "..." if len(str(message_content)) > 200 else str(message_content)
            print(f"📄 Content preview: {debug_content}")

        # Add analysis prompt
        prompt = f"""
        Extract relevant data points strictly from the above provided content for evaluating the hypothesis:
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

        print(f"🧠 Preparing to send content to LLM for analysis")
        
        # Add prompt to message content if it's a list of structured content
        if isinstance(message_content, list):
            message_content.append({"type": "text", "text": prompt})
        else:
            # If it's just text, append the prompt
            message_content += "\n\n" + prompt

        logger.debug(f"Prepared prompt for parsing")

        # Send to LLM with retry logic
        print(f"🔄 Sending to LLM API...")
        response_text = llm_provider.send_with_retry(message_content)
        logger.debug(f"Received API response for parsing")
        print(f"✅ Received response from LLM")

        # Extract JSON from response
        print(f"🔍 Extracting JSON from LLM response...")
        parsed_data = _extract_json_from_response(response_text, logger)
        logger.debug(f"Successfully parsed JSON data")
        
        # Print the extracted data for debugging
        print(f"✅ Successfully extracted JSON data")
        print(f"📊 Data preview: {json.dumps(parsed_data, indent=2)[:200]}...")
        
        return parsed_data

    except Exception as e:
        error_message = f"Error in parse_data: {str(e)}"
        logger.error(error_message, exc_info=True)
        print(f"❌ {error_message}")
        return {"error": error_message, "file": data_file}


def _prepare_file_content(data_file: str, file_type: str, logger) -> Optional[Union[List[Dict], str]]:
    """
    Prepare file content based on file type.
    
    Args:
        data_file: Path to the data file
        file_type: File extension (e.g., '.csv')
        logger: Logger instance
        
    Returns:
        Content in the format expected by the LLM provider or None if extraction fails
    """
    def truncate_content(content: str, max_length: int = 15000) -> str:
        """Truncate content to a maximum length."""
        if len(content) > max_length:
            logger.warning(f"Content length ({len(content)}) exceeds max length ({max_length}), truncating.")
            return content[:max_length] + "\n... [truncated] ..."
        return content

    try:
        print(f"🔍 Extracting content from {file_type} file")
        
        if not os.path.exists(data_file):
            logger.error(f"File not found: {data_file}")
            print(f"❌ File not found: {data_file}")
            return None
            
        if os.path.getsize(data_file) == 0:
            logger.error(f"File is empty: {data_file}")
            print(f"❌ File is empty: {data_file}")
            return None
            
        if file_type == '.csv':
            logger.debug("Processing CSV file")
            print(f"📊 Processing CSV file")
            try:
                df = pd.read_csv(data_file)
                if df.empty:
                    logger.warning(f"CSV file is empty: {data_file}")
                    print(f"⚠️ CSV file is empty")
                    return [{"type": "text", "text": "The provided CSV file is empty."}]
                content = df.to_string()
                print(f"✅ Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
                return [{"type": "text", "text": truncate_content(content)}]
            except pd.errors.EmptyDataError:
                logger.warning(f"CSV file is empty: {data_file}")
                print(f"⚠️ CSV file is empty")
                return [{"type": "text", "text": "The provided CSV file is empty."}]
            except pd.errors.ParserError as e:
                logger.error(f"Error parsing CSV: {str(e)}")
                print(f"❌ Error parsing CSV: {str(e)}")
                return [{"type": "text", "text": f"Error parsing CSV: {str(e)}"}]

        elif file_type in ['.pdf', '.PDF']:
            logger.debug("Processing PDF file")
            print(f"📑 Processing PDF file")
            # Try to determine if provider supports direct PDF handling
            try:
                with open(data_file, 'rb') as f:
                    pdf_data = base64.b64encode(f.read()).decode('utf-8')
                print(f"✅ Successfully encoded PDF (size: {len(pdf_data)} chars)")
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
                print(f"⚠️ Direct PDF handling failed, falling back to text extraction: {str(e)}")
                
                # Fall back to text extraction
                extracted_text = _extract_text_from_pdf(data_file, logger)
                if not extracted_text or extracted_text.strip() == "":
                    logger.warning(f"Failed to extract any text from PDF: {data_file}")
                    print(f"⚠️ Failed to extract any text from PDF")
                    return [{"type": "text", "text": "Failed to extract any text from the PDF file."}]
                print(f"✅ Successfully extracted text from PDF (length: {len(extracted_text)} chars)")
                return [{"type": "text", "text": truncate_content(extracted_text)}]

        elif file_type in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            logger.debug(f"Processing image file of type {file_type}")
            print(f"🖼️ Processing image file of type {file_type}")
            try:
                with open(data_file, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                media_type = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }[file_type]
                print(f"✅ Successfully encoded image (size: {len(img_data)} chars)")
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
            except Exception as e:
                logger.error(f"Error processing image file: {str(e)}")
                print(f"❌ Error processing image file: {str(e)}")
                return None

        else:
            logger.debug(f"Processing text file of type {file_type}")
            print(f"📝 Processing text file of type {file_type}")
            try:
                with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                if not content or content.strip() == "":
                    logger.warning(f"Text file is empty: {data_file}")
                    print(f"⚠️ Text file is empty")
                    return [{"type": "text", "text": "The provided text file is empty."}]
                print(f"✅ Successfully read text file (length: {len(content)} chars)")
                return [{"type": "text", "text": truncate_content(content)}]
            except UnicodeDecodeError as e:
                logger.warning(f"Unicode decode error, trying binary mode: {str(e)}")
                print(f"⚠️ Unicode decode error, trying binary mode: {str(e)}")
                # Try binary mode for potential binary files
                try:
                    with open(data_file, 'rb') as f:
                        binary_data = base64.b64encode(f.read()).decode('utf-8')
                    print(f"✅ Successfully read file in binary mode")
                    return [{"type": "text", "text": f"Binary file encoded as base64: {binary_data[:100]}..."}]
                except Exception as binary_e:
                    logger.error(f"Error reading file in binary mode: {str(binary_e)}")
                    print(f"❌ Error reading file in binary mode: {str(binary_e)}")
                    return None

    except Exception as e:
        logger.error(f"Error preparing file content: {str(e)}", exc_info=True)
        print(f"❌ Error preparing file content: {str(e)}")
        return None


def _extract_text_from_pdf(pdf_file: str, logger) -> str:
    """
    Extract text from PDF using multiple methods with fallbacks.
    
    Args:
        pdf_file: Path to the PDF file
        logger: Logger instance
        
    Returns:
        str: Extracted text or empty string if all methods fail
    """
    logger.debug("Extracting text from PDF")
    print(f"📑 Extracting text from PDF using multiple methods")
    
    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF
        logger.debug("Using PyMuPDF for text extraction")
        print(f"🔍 Trying PyMuPDF for text extraction")
        doc = fitz.open(pdf_file)
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += page_text
            print(f"  - Page {page_num+1}: Extracted {len(page_text)} characters")
        if text:
            print(f"✅ Successfully extracted text using PyMuPDF: {len(text)} characters")
            return text
        print(f"⚠️ PyMuPDF did not extract any text, trying other methods")
    except ImportError:
        logger.warning("PyMuPDF not installed, skipping")
        print(f"⚠️ PyMuPDF not installed, skipping")
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {str(e)}, trying pdfplumber")
        print(f"⚠️ PyMuPDF extraction failed: {str(e)}, trying pdfplumber")
    
    # Try pdfplumber next
    try:
        import pdfplumber
        logger.debug("Using pdfplumber for text extraction")
        print(f"🔍 Trying pdfplumber for text extraction")
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text += page_text
                print(f"  - Page {page_num+1}: Extracted {len(page_text)} characters")
        if text:
            print(f"✅ Successfully extracted text using pdfplumber: {len(text)} characters")
            return text
        print(f"⚠️ pdfplumber did not extract any text, trying other methods")
    except ImportError:
        logger.warning("pdfplumber not installed, skipping")
        print(f"⚠️ pdfplumber not installed, skipping")
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {str(e)}, trying textract")
        print(f"⚠️ pdfplumber extraction failed: {str(e)}, trying textract")
    
    # Try textract as last resort
    try:
        import textract
        logger.debug("Using textract for text extraction")
        print(f"🔍 Trying textract for text extraction")
        text = textract.process(pdf_file, method='pdfminer').decode('utf-8')
        if text:
            print(f"✅ Successfully extracted text using textract: {len(text)} characters")
            return text
        print(f"⚠️ textract did not extract any text")
    except ImportError:
        logger.warning("textract not installed, skipping")
        print(f"⚠️ textract not installed, skipping")
    except Exception as e:
        logger.error(f"textract extraction failed: {str(e)}")
        print(f"❌ textract extraction failed: {str(e)}")
    
    # If all methods failed
    logger.error("All PDF extraction methods failed")
    print(f"❌ All PDF extraction methods failed")
    return ""


def _extract_json_from_response(response_text: str, logger) -> Dict:
    """
    Extract JSON data from LLM response.
    
    Args:
        response_text: Text response from LLM
        logger: Logger instance
        
    Returns:
        Dict: Parsed JSON data or error information
    """
    try:
        if not response_text:
            logger.error("Empty response from LLM")
            print(f"❌ Empty response from LLM")
            return {"extraction_error": "Empty response from LLM"}
            
        # Try to extract JSON using autogen
        print(f"🔍 Extracting JSON using code blocks")
        extracted_blocks = extract_code(response_text)

        # Look for JSON blocks
        json_str = None
        for lang, block in extracted_blocks:
            if lang.lower() in ['json', '']:
                try:
                    # Try to parse as JSON to validate
                    parsed = json.loads(block)
                    json_str = block
                    print(f"✅ Found valid JSON code block")
                    break
                except json.JSONDecodeError:
                    continue

        # If no valid JSON block found, try parsing the whole response
        if not json_str:
            logger.warning("No JSON code block found, trying to parse entire response")
            print(f"⚠️ No JSON code block found, trying to parse entire response")
            try:
                parsed = json.loads(response_text)
                json_str = response_text
                print(f"✅ Successfully parsed entire response as JSON")
            except json.JSONDecodeError:
                # Try to find JSON in response using regex
                import re
                logger.warning("Failed to parse response as JSON, trying regex extraction")
                print(f"⚠️ Failed to parse response as JSON, trying regex extraction")
                json_pattern = r'({[\s\S]*})'
                match = re.search(json_pattern, response_text)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                        json_str = match.group(1)
                        print(f"✅ Successfully extracted JSON using regex")
                    except json.JSONDecodeError:
                        logger.error("Regex extraction failed to find valid JSON")
                        print(f"❌ Regex extraction failed to find valid JSON")
                        return {
                            "extraction_error": "Failed to parse LLM response",
                            "raw_response": response_text[:500] + ("..." if len(response_text) > 500 else "")
                        }
                else:
                    logger.error("Failed to parse response as JSON")
                    print(f"❌ Failed to parse response as JSON")
                    return {
                        "extraction_error": "Failed to parse LLM response",
                        "raw_response": response_text[:500] + ("..." if len(response_text) > 500 else "")
                    }

        parsed_data = json.loads(json_str)
        return parsed_data

    except Exception as e:
        logger.error(f"Error extracting JSON from response: {str(e)}", exc_info=True)
        print(f"❌ Error extracting JSON from response: {str(e)}")
        return {
            "extraction_error": f"Error: {str(e)}",
            "raw_response": response_text[:500] + ("..." if len(response_text) > 500 else "")
        }


# Standalone version for direct use without InFactNode
def parse_standalone(file_path: str, hypothesis: str, provider: str, model: str, api_key: str) -> Dict:
    """
    Parse file data and extract information relevant to a hypothesis using the specified LLM.
    Standalone version for use without requiring an InFactNode instance.
    
    Args:
        file_path: Path to the file to be analyzed
        hypothesis: The hypothesis to evaluate against
        provider: "GPT" or "Anthropic" or "DeepSeek"
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet")
        api_key: API key for the provider
        
    Returns:
        Dict containing parsed data or error information
    """
    # Setup simple logger
    logger = logging.getLogger("parse_standalone")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    logger.info(f"Parsing file: {file_path} using {provider}/{model}")
    print(f"🚀 Starting parse_standalone for {file_path} using {provider}/{model}")
    
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return {"error": error_msg}
        
    file_type = Path(file_path).suffix.lower()
    
    try:
        # Import providers dynamically to avoid circular imports
        print(f"🔌 Initializing {provider} provider with model {model}")
        if provider.lower() == "gpt" or provider.lower() == "openai":
            from ..providers.openai_provider import OpenAIProvider
            llm_provider = OpenAIProvider(api_key=api_key, model=model)
        elif provider.lower() == "anthropic":
            from ..providers.anthropic_provider import AnthropicProvider
            llm_provider = AnthropicProvider(api_key=api_key, model=model)
        else:
            error_msg = f"Unsupported provider: {provider}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return {"error": error_msg}
            
        # Use the core parse_data function with the appropriate provider
        print(f"🧠 Starting data parsing process")
        result = parse_data(file_path, hypothesis, llm_provider, logger)
        print(f"✅ Parsing completed")
        return result
        
    except Exception as e:
        error_msg = f"Error in parse_standalone: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}