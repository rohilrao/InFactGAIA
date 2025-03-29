import json
import base64
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging
from autogen.code_utils import extract_code


def parse_data(data_file: str, hypothesis: str, llm_provider, logger) -> Dict:
    """
    Parse different file types using LLM assistance.
    
    Args:
        data_file: Path to the data file
        hypothesis: The hypothesis being evaluated
        llm_provider: LLM provider instance
        logger: Logger instance
        
    Returns:
        Dict: Parsed data
    """
    logger.info(f"Parsing data file: {data_file}")
    file_type = Path(data_file).suffix.lower()

    try:
        # Prepare content based on file type
        message_content = _prepare_file_content(data_file, file_type, logger)
        
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

        logger.debug(f"Prepared prompt for parsing")

        # Send to LLM with retry logic
        response_text = llm_provider.send_with_retry(message_content)
        logger.debug(f"Received API response for parsing")

        # Extract JSON from response
        parsed_data = _extract_json_from_response(response_text, logger)
        logger.debug(f"Successfully parsed JSON data")
        
        return parsed_data

    except Exception as e:
        logger.error(f"Error in parse_data: {str(e)}", exc_info=True)
        raise


def _prepare_file_content(data_file: str, file_type: str, logger) -> Any:
    """
    Prepare file content based on file type.
    
    Args:
        data_file: Path to the data file
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
            df = pd.read_csv(data_file)
            content = df.to_string()
            return [{"type": "text", "text": truncate_content(content)}]

        elif file_type in ['.pdf', '.PDF']:
            logger.debug("Processing PDF file")
            with open(data_file, 'rb') as f:
                pdf_data = base64.b64encode(f.read()).decode('utf-8')
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

        elif file_type in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            logger.debug(f"Processing image file of type {file_type}")
            with open(data_file, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
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
            with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return [{"type": "text", "text": truncate_content(content)}]

    except Exception as e:
        logger.error(f"Error preparing file content: {str(e)}", exc_info=True)
        raise


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