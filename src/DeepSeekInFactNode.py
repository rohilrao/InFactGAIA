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


def extract_text_from_pdf(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

class DeepSeekInFactNode:
    def __init__(self,
                hypothesis: str,
                api_key: str,
                model: str,
                prior_log_odds: float = 0.0,
                log_level: int = logging.INFO):
    
      """Initialize node with logging configuration."""
      # Setup logging
      self.logger = logging.getLogger(__name__)
      self.logger.setLevel(log_level)

      # Create a unique log file for this instance
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      log_dir = Path("logs")
      log_dir.mkdir(exist_ok=True)

      file_handler = logging.FileHandler(
          log_dir / f"infact_{timestamp}.log",
          encoding='utf-8'
      )
      formatter = logging.Formatter(
          '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
      )
      file_handler.setFormatter(formatter)
      self.logger.addHandler(file_handler)

      # Initialize API client
      self.api_key = api_key
      self.model = model
      self.logger.info(f"Initializing DeepSeekInFactNode with hypothesis: {hypothesis}")
      self.client = OpenAI(api_key= api_key, base_url="https://api.deepseek.com")

      # Store parameters
      self.hypothesis = hypothesis
      self.prior_log_odds = prior_log_odds
      self.current_posterior = prior_log_odds
      self.data_points = []

      self.logger.info("Initialization complete")

    def save(self, filename: str):
        """Save node's data to a JSON file, preserving existing data and logging progress."""
        
        node_state_exists = os.path.exists(filename)
        
        if node_state_exists:
            try:
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
                    print(f"🔄 Existing node state found in {filename}. Updating it.")
            except json.JSONDecodeError:
                existing_data = {}
                print(f"⚠️ Node state file {filename} is corrupted. Resetting state.")
        else:
            existing_data = {}
            print(f"✨ No existing node state found. Creating a fresh node state at {filename}.")

        data = {
            'hypothesis': self.hypothesis,
            'prior_log_odds': self.prior_log_odds,
            'current_posterior': self.current_posterior,
            'data_points': [
                {
                    'metadata': dp['metadata'],
                    'raw_data': dp['raw_data'],
                    'l_plus': dp['l_plus'],
                    'l_minus': dp['l_minus'],
                    'posterior': dp['posterior'],
                    'confidence_assessment': dp.get('confidence_assessment', {}),
                    'analysis_rationale': dp.get('analysis_rationale', '')
                }
                for dp in self.data_points
            ]
        }

        # ✅ Merge new data with old (avoiding duplication)
        existing_data.update(data)

        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)

        print(f"✅ Node state successfully saved to {filename}.")
        self.logger.info(f"Saving node data to {filename}")


    @classmethod
    def load(cls, filename: str, model: str, api_key: str = None):
        """Load node from a JSON file with logging to indicate if an existing state is found."""
        
        if not os.path.exists(filename):
            print(f"🚫 No existing node state found at {filename}. Creating a new node.")
            return cls(hypothesis="", api_key=api_key, model=model, prior_log_odds=0)  # Default values

        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"🔄 Loading existing node state from {filename}.")
        except json.JSONDecodeError:
            print(f"⚠️ Error: Corrupted node state file {filename}. Creating a fresh node.")
            return cls(hypothesis="", api_key=api_key, model=model, prior_log_odds=0)  # Default values

        # Create new node
        node = cls(
            hypothesis=data.get('hypothesis', ""),
            api_key=api_key,
            model=model,
            prior_log_odds=data.get('prior_log_odds', 0)
        )

        # Restore state
        node.current_posterior = data.get('current_posterior', 0)
        node.data_points = data.get('data_points', [])

        print(f"✅ Successfully loaded node state from {filename}.")
        node.logger.info(f"Loaded node data from {filename}")
        
        return node

    def process_data(self, data_file: str) -> Tuple[float, Tuple[float, float]]:
        """Process a new data file and update beliefs."""
        self.logger.info(f"Processing data file: {data_file}")

        try:
            # Parse data
            parsed_data = self._parse_data(data_file)
            self.logger.debug(f"Parsed data: {json.dumps(parsed_data, indent=2)}")

            # Check redundancy
            if self._is_redundant(parsed_data):
                self.logger.info("Data determined to be redundant, skipping")
                return self.current_posterior, self._calculate_uncertainty()

            # Analyze data
            l_plus, l_minus, code = self._analyze_data(parsed_data)
            self.logger.info(f"Analysis results - l_plus: {l_plus}, l_minus: {l_minus}")

            # Update posterior
            new_posterior = self.current_posterior + l_plus - l_minus
            self.logger.info(f"Updated posterior from {self.current_posterior} to {new_posterior}")

            # Store data point
            self.data_points.append({
                'raw_data': parsed_data,
                'metadata': self._extract_metadata(data_file),
                'l_plus': l_plus,
                'l_minus': l_minus,
                'posterior': new_posterior,
                'confidence_assessment': parsed_data.get('confidence_assessment', {
                    'confidence_score': 0,
                    'explanation': 'No confidence assessment available',
                    'key_strengths': [],
                    'key_limitations': []
                }),
                'analysis_rationale': code  # Store the analysis code used
            })

            self.current_posterior = new_posterior
            lower, upper = self._calculate_uncertainty()

            self.logger.info(f"Processing complete. Current probability: {self._to_probability(new_posterior):.2%} ({lower:.2%}, {upper:.2%})")
            return new_posterior, (lower, upper)

        except Exception as e:
            self.logger.error(f"Error processing {data_file}: {str(e)}", exc_info=True)
            raise


    def _parse_data(self, data_file: str) -> Dict:
        """Parse different file types using LLM assistance."""
        self.logger.info(f"Parsing data file: {data_file}")
        file_type = Path(data_file).suffix.lower()

        try:
            if file_type == '.csv':
                self.logger.debug("Processing CSV file")
                df = pd.read_csv(data_file)
                content = df.to_string()
                message_content = [{"type": "text", "text": content}]

            elif file_type in ['.pdf', '.PDF']:
                self.logger.debug("Processing PDF file")
                extracted_text = extract_text_from_pdf(data_file)
                message_content = [{"type": "text", "text": extracted_text}]

            elif file_type in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                self.logger.debug(f"Processing image file of type {file_type}")
                with open(data_file, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                media_type = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
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
                self.logger.debug(f"Processing text file of type {file_type}")
                with open(data_file, 'r') as f:
                    content = f.read()
                message_content = [{"type": "text", "text": content}]

            # Add analysis prompt
            prompt = f"""
            Given this data:
            {message_content}

            Extract relevant data points for evaluating the hypothesis:
            "{self.hypothesis}"

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

            self.logger.debug(f"Prepared prompt: {prompt}")

            # Send to GPT API
            self.logger.info("Sending request to DeepSeek API")

            message = self.client.chat.completions.create(
                model= self.model, #"gpt-4o-mini-2024-07-18",
                max_completion_tokens = 8192,
                messages=[
                  {"role": "user", 
                   "content": prompt}
                ]
                
              )

            # Extract and parse response
            response_text = self._get_message_text(message)
            self.logger.debug(f"Received API response: {response_text}")

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
                self.logger.warning("No JSON code block found, trying to parse entire response")
                try:
                    parsed = json.loads(response_text)
                    json_str = response_text
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse response as JSON")
                    return {
                        "extraction_error": "Failed to parse LLM response",
                        "raw_response": response_text
                    }

            parsed_data = json.loads(json_str)
            self.logger.debug(f"Successfully parsed JSON data: {json.dumps(parsed_data, indent=2)}")
            return parsed_data

        except Exception as e:
            self.logger.error(f"Error in _parse_data: {str(e)}", exc_info=True)
            raise


    def _extract_metadata(self, data_file: str) -> Dict[str, Any]:
        """Extract metadata from the data file."""
        self.logger.info(f"Extracting metadata from {data_file}")

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
                        if pdf.metadata:
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
                    self.logger.warning("PyPDF2 not installed, skipping PDF metadata extraction")

            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        basic_metadata.update({
                            "image_format": img.format,
                            "image_size": img.size,
                            "image_mode": img.mode,
                            "image_info": dict(img.info)
                        })
                except ImportError:
                    self.logger.warning("Pillow not installed, skipping image metadata extraction")

            elif file_path.suffix.lower() == '.html':
                try:
                    from bs4 import BeautifulSoup
                    with open(file_path, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                        meta_tags = {}
                        for meta in soup.find_all('meta'):
                            name = meta.get('name', meta.get('property', ''))
                            content = meta.get('content', '')
                            if name and content:
                                meta_tags[name] = content

                        basic_metadata.update({
                            "title": soup.title.string if soup.title else '',
                            "meta_tags": meta_tags,
                            "has_article": bool(soup.find('article')),
                            "has_main": bool(soup.find('main')),
                            "num_headers": len(soup.find_all(['h1', 'h2', 'h3'])),
                            "has_tables": bool(soup.find_all('table'))
                        })
                except ImportError:
                    self.logger.warning("BeautifulSoup4 not installed, skipping HTML metadata extraction")

            elif file_path.suffix.lower() == '.csv':
                try:
                    df = pd.read_csv(file_path)
                    basic_metadata.update({
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "column_names": list(df.columns),
                        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "has_nulls": df.isnull().any().any()
                    })
                except Exception as e:
                    self.logger.warning(f"Error extracting CSV metadata: {str(e)}")

            self.logger.debug(f"Extracted metadata: {json.dumps(basic_metadata, indent=2)}")
            return basic_metadata

        except Exception as e:
            self.logger.error(f"Error in _extract_metadata: {str(e)}", exc_info=True)
            return {
                "filename": data_file,
                "error": str(e)
            }

    def _analyze_data(self, data: Dict) -> Tuple[float, float, str]:
        """Generate and execute analysis code using LLM."""
        self.logger.info("Analyzing parsed data")

        MAX_LOG_LIKELIHOOD_RATIO = 5.

        try:
            # Generate analysis code
            prompt = f"""
            Given this data:
            {json.dumps(data, indent=2)}

            Generate Python code to calculate log likelihoods for the hypothesis:
            "{self.hypothesis}"

            This should be a single function named `calculate_log_likelihoods`.
            It should take a single argument, a dict with the format given above,
            and output only the tuple of log-likelihoods
              l_plus = log P(data | hypothesis),
              l_minus = log P(data | not hypothesis).

            The code should:
            1. Calculate l_plus and l_minus (log likelihoods)
            2. Handle uncertainties properly
            3. Account for data quality and potential biases
            4. Limit overconfidence by capping the absolute difference between l_plus and l_minus to {MAX_LOG_LIKELIHOOD_RATIO}.
            4. Use the usual libraries such as numpy and scipy for calculations
            5. Use print() to output intermediate results, as well as the final result before returning.
            6. Ensure numerical stability by avoiding edge cases such as infinity (`inf`) and NaN values in likelihood and probability calculations.

            Return only executable Python code with the function definition.
            Do not include the function call itself.
            """
            
            
            self.logger.debug(f"Analysis prompt: {prompt}")
            
            message = self.client.chat.completions.create(
                model= self.model, #"gpt-4o-mini-2024-07-18",
                max_completion_tokens = 8192,
                messages=[
                  {"role": "user", 
                   "content": prompt}
                ]
                
              )

            response_text = self._get_message_text(message)
            self.logger.debug(f"API response: {response_text}")

            # Extract code using autogen
            from autogen.code_utils import extract_code
            extracted_code = extract_code(response_text)

            if not extracted_code:
                self.logger.error("No code block found in API response")
                raise

            # Get the first Python code block
            code = None
            for lang, code_block in extracted_code:
                if lang.lower() in ['python', 'py', '']:
                    code = code_block
                    break

            if not code:
                self.logger.error("No Python code block found in API response")
                raise

            self.logger.debug(f"Extracted Python code: {code}")

            # Execute the code
            l_plus, l_minus, code = self._execute_code_with_debug(code, data)
            return l_plus, l_minus, code

        except Exception as e:
            self.logger.error(f"Error in _analyze_data: {str(e)}", exc_info=True)
            raise


    def _execute_code_with_debug(self, code: str, data: Dict, max_attempts: int = 5) -> Tuple[float, float, str]:
        """Execute code with debug loop for error correction."""
        globals_dict = {
            "np": np,
            "math": math,
            "data": data
        }

        attempt = 1
        while attempt <= max_attempts:
            self.logger.info(f"Code execution attempt {attempt}/{max_attempts}")
            self.logger.debug(f"Executing code:\n{code}")

            try:
                exec(code, globals_dict)
                l_plus, l_minus = globals_dict['calculate_log_likelihoods'](data)

                # Validate outputs
                if l_plus is None or l_minus is None:
                    raise ValueError("Code did not define l_plus and l_minus")

                if not (isinstance(l_plus, (int, float)) and isinstance(l_minus, (int, float))):
                    raise ValueError("l_plus and l_minus must be numeric values")

                self.logger.info(f"Code execution successful - l_plus: {l_plus}, l_minus: {l_minus}")
                return float(l_plus), float(l_minus), code

            except Exception as e:
                self.logger.warning(f"Code execution failed on attempt {attempt}: {str(e)}")

                if attempt == max_attempts:
                    self.logger.error("Max attempts reached, raising error")
                    raise RuntimeError(f"Failed to generate working code after {max_attempts} attempts. Final error: {str(e)}")

                # Ask LLM to fix the code
                debug_prompt = f"""
                The following code failed with error: {str(e)}

                Code:
                ```python
                {code}
                ```

                Input data:
                ```json
                {json.dumps(data, indent=2)}
                ```

                Please fix the code to:
                1. Handle the error properly
                2. Return numeric values for l_plus and l_minus
                3. Include proper error checking
                4. Handle edge cases in the input data

                Return only the corrected Python code.
                """

                self.logger.debug(f"Sending debug prompt to LLM:\n{debug_prompt}")

                message = self.client.chat.completions.create(
                model= self.model, #"gpt-4o-mini-2024-07-18",
                max_completion_tokens = 8192,
                messages=[
                  {"role": "user", 
                   "content": debug_prompt}
                  ]
                
                   ) 

                # Extract corrected code
                response_text = self._get_message_text(message)
                extracted_code = extract_code(response_text)

                if not extracted_code:
                    self.logger.error("No code block found in debug response")
                    attempt += 1
                    continue

                # Get the first Python code block
                for lang, code_block in extracted_code:
                    if lang.lower() in ['python', 'py', '']:
                        code = code_block
                        break
                else:
                    self.logger.error("No Python code block found in debug response")
                    attempt += 1
                    continue

            attempt += 1

        # Should never reach here due to raise in loop
        raise RuntimeError("Unexpected error in debug loop")

    def _is_redundant(self, new_data: Dict) -> bool:
        """Check if new data is redundant with existing data."""
        self.logger.info("Checking for data redundancy")

        if not self.data_points:
            self.logger.debug("No existing data points, not redundant")
            return False

        try:
            prompt = f"""
            Compare the following new data:
            {json.dumps(new_data, indent=2)}

            With these existing data points:
            {json.dumps([dp['raw_data'] for dp in self.data_points], indent=2)}

            Is the new data redundant with any existing data points?
            Consider:
            1. Same source or study being cited
            2. Same measurements within uncertainty
            3. Derived results from already incorporated primary data

            Return "true" if redundant, "false" if novel information.
            """

            self.logger.debug(f"Redundancy check prompt: {prompt}")

            
            message = self.client.chat.completions.create(
                model= self.model, #"gpt-4o-mini-2024-07-18",
                max_completion_tokens = 8192,
                messages=[
                  {"role": "user", 
                   "content": prompt}
                  ]
                   ) 

            response = self._get_message_text(message)
            self.logger.debug(f"Redundancy check response: {response}")

            is_redundant = response.strip().lower() == "true"
            self.logger.info(f"Redundancy check result: {is_redundant}")
            return is_redundant

        except Exception as e:
            self.logger.error(f"Error in _is_redundant: {str(e)}", exc_info=True)
            raise

    def _get_message_text(self, message) -> str:
      """Extract text content from an OpenAI API response."""
      if message and hasattr(message, 'choices') and len(message.choices) > 0:
          content_block = message.choices[0].message
          if hasattr(content_block, 'content') and isinstance(content_block.content, str):
              return content_block.content
      print("Error: No content found in API response")
      return ""


    def _calculate_uncertainty(self) -> tuple[float, float]:
        """Calculate 95% confidence interval for the posterior probability.

        Returns:
            tuple[float, float]: Lower and upper bounds of the 95% CI
        """
        # Convert current posterior log-odds to probability
        p = self._to_probability(self.current_posterior)

        # Calculate total weight of evidence from Bayes factors
        total_evidence = sum(
            abs(math.exp(dp['l_plus'] - dp['l_minus']) - 1)
            for dp in self.data_points
        )

        if total_evidence < 1e-6:
            return (0.0, 1.0)  # Default CI for effectively no data

        # Each Bayes factor represents the weight of evidence
        # The concentration parameter of our Beta should reflect this
        concentration = total_evidence

        # Calculate Beta parameters to maintain the mean at p
        alpha = concentration * p
        beta = concentration * (1 - p)

        # Calculate 95% confidence interval
        from scipy import stats
        ci_low, ci_high = stats.beta.interval(0.95, alpha, beta)

        # Clip to [0, 1]
        ci_low = max(0.0, min(1.0, ci_low))
        ci_high = max(0.0, min(1.0, ci_high))

        return ci_low, ci_high

    @staticmethod
    def _to_probability(log_odds: float) -> float:
        """Convert log odds to probability."""
        return 1 / (1 + np.exp(-log_odds))