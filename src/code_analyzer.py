import json
from typing import Dict, Tuple
import numpy as np
from autogen.code_utils import extract_code
from infact_utils import call_llm


def analyze_data(data: Dict, hypothesis: str, credentials: Tuple[str, ...]) -> Tuple[float, float, str]:
    """
    Generate and execute analysis code using LLM.
    
    Args:
        data: Parsed data dictionary
        hypothesis: The hypothesis being evaluated
        credentials: Tuple of strings containing authentication credentials
        
    Returns:
        Tuple: (l_plus, l_minus, code) - log likelihoods and the analysis code
    """
    print("Analyzing parsed data")
    
    # Initialize LLM provider with credentials
    provider, model, api_key = credentials

    MAX_LOG_LIKELIHOOD_RATIO = 5.0

    try:
        # Generate analysis code
        prompt = f"""
        Given this data:
        {json.dumps(data, indent=2)}

        Generate Python code to calculate log likelihoods for the hypothesis:
        "{hypothesis}"

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
        5. Use the usual libraries such as numpy and scipy for calculations
        6. Use print() to output intermediate results, as well as the final result before returning.
        7. Ensure numerical stability by avoiding edge cases such as infinity (`inf`) and NaN values in likelihood and probability calculations.

        Return only executable Python code with the function definition.
        Do not include the function call itself.
        """

        print("Sending analysis prompt to LLM")
        response_text = call_llm(provider, api_key, model, prompt)
        print("Received analysis code from LLM")

        # Extract code using autogen
        extracted_code = extract_code(response_text)

        if not extracted_code:
            print("ERROR: No code block found in API response")
            raise ValueError("No code block found in LLM response")

        # Get the first Python code block
        code = None
        for lang, code_block in extracted_code:
            if lang.lower() in ['python', 'py', '']:
                code = code_block
                break

        if not code:
            print("ERROR: No Python code block found in API response")
            raise ValueError("No Python code block found in LLM response")

        print("Extracted Python code")

        # Execute the code
        l_plus, l_minus = _execute_code_with_debug(code, data, credentials)
        return l_plus, l_minus, code

    except Exception as e:
        print(f"ERROR in analyze_data: {str(e)}")
        raise


def _execute_code_with_debug(code: str, data: Dict, credentials: Tuple[str, ...], max_attempts: int = 5) -> Tuple[float, float]:
    """
    Execute code with debug loop for error correction.
    
    Args:
        code: Python code to execute
        data: Input data for the code
        credentials: Tuple of strings containing authentication credentials
        max_attempts: Maximum number of debugging attempts
        
    Returns:
        Tuple: (l_plus, l_minus) - calculated log likelihoods
    """
    globals_dict = {
        "np": np,
        "math": __import__('math'),
        "data": data
    }

    provider, model, api_key = credentials
    
    attempt = 1
    while attempt <= max_attempts:
        print(f"Code execution attempt {attempt}/{max_attempts}")
        print(f"Executing code:\n{code}")

        try:
            exec(code, globals_dict)
            l_plus, l_minus = globals_dict['calculate_log_likelihoods'](data)

            # Validate outputs
            if l_plus is None or l_minus is None:
                raise ValueError("Code did not define l_plus and l_minus")

            if not (isinstance(l_plus, (int, float)) and isinstance(l_minus, (int, float))):
                raise ValueError("l_plus and l_minus must be numeric values")

            print(f"Code execution successful - l_plus: {l_plus}, l_minus: {l_minus}")
            return float(l_plus), float(l_minus)

        except Exception as e:
            print(f"WARNING: Code execution failed on attempt {attempt}: {str(e)}")

            if attempt == max_attempts:
                print("ERROR: Max attempts reached, raising error")
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

            print(f"Sending debug prompt to LLM")
            response_text = call_llm(provider, api_key, model, debug_prompt)

            # Extract corrected code
            extracted_code = extract_code(response_text)

            if not extracted_code:
                print("ERROR: No code block found in debug response")
                attempt += 1
                continue

            # Get the first Python code block
            for lang, code_block in extracted_code:
                if lang.lower() in ['python', 'py', '']:
                    code = code_block
                    break
            else:
                print("ERROR: No Python code block found in debug response")
                attempt += 1
                continue

        attempt += 1

    # Should never reach here due to raise in loop
    raise RuntimeError("Unexpected error in debug loop")
