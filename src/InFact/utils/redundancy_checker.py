import json
from typing import Dict, List
import logging


def is_redundant(new_data: Dict, existing_data_points: List[Dict], llm_provider, logger) -> bool:
    """
    Check if new data is redundant with existing data.
    
    Args:
        new_data: Newly parsed data
        existing_data_points: List of previously processed data points
        llm_provider: LLM provider instance
        logger: Logger instance
        
    Returns:
        bool: True if data is redundant, False otherwise
    """
    logger.info("Checking for data redundancy")

    if not existing_data_points:
        logger.debug("No existing data points, not redundant")
        return False

    try:
        # Extract raw data from existing data points
        existing_raw_data = [dp['raw_data'] for dp in existing_data_points]
        
        prompt = f"""
        Compare the following new data:
        {json.dumps(new_data, indent=2)}

        With these existing data points:
        {json.dumps(existing_raw_data, indent=2)}

        Is the new data redundant with any existing data points?
        Consider:
        1. Same source or study being cited
        2. Same measurements within uncertainty
        3. Derived results from already incorporated primary data

        Return "true" if redundant, "false" if novel information.
        """

        logger.debug(f"Sending redundancy check prompt to LLM")
        response = llm_provider.send_message(prompt)
        logger.debug(f"Received redundancy check response")

        is_redundant = response.strip().lower() == "true"
        logger.info(f"Redundancy check result: {is_redundant}")
        return is_redundant

    except Exception as e:
        logger.error(f"Error in is_redundant: {str(e)}", exc_info=True)
        # Default to not redundant in case of error
        logger.warning("Error in redundancy check, defaulting to not redundant")
        return False