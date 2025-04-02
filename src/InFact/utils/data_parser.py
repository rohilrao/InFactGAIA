import json
import base64
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List
import logging
from autogen.code_utils import extract_code
import os


def parse_data(data_file: str, hypothesis: str, llm_provider, logger) -> Dict:
    """
    Parse different file types using LLM assistance.
    """
    print("[parse_data] Starting to parse data file:", data_file)
    file_type = Path(data_file).suffix.lower()
    
    try:
        # Prepare content based on file type
        print(f"[parse_data] Calling _prepare_file_content for file: {data_file}")
        message_content = _prepare_file_content(data_file, file_type, logger)
        
        print(f"[parse_data] Message Content Type: {type(message_content)}")
        if isinstance(message_content, list):
            print(f"[parse_data] Message Content List Length: {len(message_content)}")
            for i, item in enumerate(message_content):
                print(f"[parse_data] Item {i} type: {item.get('type', 'unknown')}")
                if item.get('type') == 'text' and 'text' in item:
                    text_sample = item['text'][:100] + "..." if len(item['text']) > 100 else item['text']
                    print(f"[parse_data] Text content sample: {text_sample}")

        # Add analysis prompt
        print("[parse_data] Creating analysis prompt")
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

        # Add prompt to message content if it's a list of structured content
        if isinstance(message_content, list):
            print("[parse_data] Adding prompt to message_content list")
            message_content.append({"type": "text", "text": prompt})
        else:
            # If it's just text, append the prompt
            print("[parse_data] Adding prompt to message_content string")
            message_content += "\n\n" + prompt

        # Print final message structure before sending to LLM
        print(f"[parse_data] Final message type before sending to LLM: {type(message_content)}")
        if isinstance(message_content, list):
            print(f"[parse_data] Final message list length: {len(message_content)}")
            for i, item in enumerate(message_content):
                print(f"[parse_data] Final message item {i} type: {item.get('type', 'unknown')}")

        # Send to LLM with retry logic
        print("[parse_data] Sending message to LLM provider")
        response_text = llm_provider.send_with_retry(message_content)
        print(f"[parse_data] Received API response (first 100 chars): {response_text[:100]}...")

        # Extract JSON from response
        print("[parse_data] Extracting JSON from response")
        parsed_data = _extract_json_from_response(response_text, logger)
        print("[parse_data] Successfully parsed JSON data")
        
        return parsed_data

    except Exception as e:
        print(f"[parse_data] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def _prepare_file_content(data_file: str, file_type: str, logger) -> Any:
    """
    Prepare file content based on file type.
    """
    print(f"[_prepare_file_content] Processing file: {data_file} with type: {file_type}")
    
    def truncate_content(content: str, max_length: int = 15000) -> str:
        """Truncate content to a maximum length."""
        if len(content) > max_length:
            print(f"[_prepare_file_content] Content length ({len(content)}) exceeds max length ({max_length}), truncating.")
            return content[:max_length] + "\n... [truncated] ..."
        return content

    try:
        if file_type == '.csv':
            print("[_prepare_file_content] Processing CSV file")
            df = pd.read_csv(data_file)
            content = df.to_string()
            print(f"[_prepare_file_content] CSV content length: {len(content)}")
            return [{"type": "text", "text": truncate_content(content)}]

        elif file_type in ['.pdf', '.PDF']:
            print("[_prepare_file_content] Processing PDF file")
            # Rest of PDF handling code...
            # (existing code)
            
        elif file_type in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
            print(f"[_prepare_file_content] Processing image file of type {file_type}")
            # Rest of image handling code...
            # (existing code)

        else:
            print(f"[_prepare_file_content] Processing text file of type {file_type}")
            try:
                with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                print(f"[_prepare_file_content] Text file content length: {len(content)}")
                return [{"type": "text", "text": truncate_content(content)}]
            except Exception as file_error:
                print(f"[_prepare_file_content] Error reading file: {str(file_error)}")
                raise

    except Exception as e:
        print(f"[_prepare_file_content] Error preparing file content: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def parse_standalone(file_path: str, hypothesis: str, provider: str, model: str, api_key: str) -> Dict:
    """
    Parse file data and extract information relevant to a hypothesis using the specified LLM.
    Standalone version for use without requiring an InFactNode instance.
    """
    print(f"[parse_standalone] Starting parse_standalone for file: {file_path}")
    print(f"[parse_standalone] Using provider: {provider}, model: {model}")
    print(f"[parse_standalone] Hypothesis: {hypothesis}")
    
    # Setup simple logger
    logger = logging.getLogger("parse_standalone")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    file_type = Path(file_path).suffix.lower()
    print(f"[parse_standalone] File type detected: {file_type}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"[parse_standalone] ERROR: File does not exist: {file_path}")
            return {"error": f"File does not exist: {file_path}"}
            
        # Import providers dynamically to avoid circular imports
        print(f"[parse_standalone] Initializing LLM provider: {provider}")
        if provider.lower() == "gpt" or provider.lower() == "openai":
            print("[parse_standalone] Importing OpenAI provider")
            from ..providers.openai_provider import OpenAIProvider
            llm_provider = OpenAIProvider(api_key=api_key, model=model)
            print("[parse_standalone] OpenAI provider initialized")
        elif provider.lower() == "anthropic":
            print("[parse_standalone] Importing Anthropic provider")
            from ..providers.anthropic_provider import AnthropicProvider
            llm_provider = AnthropicProvider(api_key=api_key, model=model)
            print("[parse_standalone] Anthropic provider initialized")
        else:
            print(f"[parse_standalone] ERROR: Unsupported provider: {provider}")
            return {"error": f"Unsupported provider: {provider}"}
            
        # Use the core parse_data function with the appropriate provider
        print("[parse_standalone] Calling parse_data function")
        result = parse_data(file_path, hypothesis, llm_provider, logger)
        print("[parse_standalone] Received result from parse_data")
        return result
        
    except Exception as e:
        print(f"[parse_standalone] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}