import streamlit as st
import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import utility modules
from page_utils.step_utils import initialize_session_state, show_step_progress
from page_utils.db_utils import initialize_db
from page_utils.step1_setup import display_setup_step
from page_utils.step2_hypothesis_setup import display_combined_hypothesis_step
from page_utils.step3_add_evidence import display_file_upload_step
from page_utils.step4_code_review import display_code_review_step
from page_utils.evidence_step import display_evidence_step
from page_utils.step5_results import display_results_step


# Import other necessary modules for remaining steps
from anthropic import Anthropic
import openai
import datetime
import json
import tempfile
from pathlib import Path
import pandas as pd
from bson.objectid import ObjectId

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds "src" to path

from utils import parse_data 
from jinja2 import Template


# Initialize session state for tracking progress
initialize_session_state()

# Show step progress bar
show_step_progress()

# Initialize database connections
client, db, fs, hypothesis_collection = initialize_db()

# Helper function for consistent ObjectId handling
def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except:
            return id_value
    return id_value

# ------------------------------------------------
# Helper: LLM call function
# ------------------------------------------------
def call_llm(provider, model, api_key, prompt_text):
    """
    Function to call GPT or Anthropic.
    """
    # For GPT
    if provider == "GPT":
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.1
        )
        return response.choices[0].message.content

    # For Anthropic
    elif provider == "Anthropic":
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        return message.content[0].text

    else:
        raise ValueError("Unsupported provider")

# Step 1: AI Model Config
if st.session_state.process_step == 1:
    if display_setup_step():
        st.session_state.process_step = 2
        st.rerun()

#Step 2
# Replace the individual step calls with this:
elif st.session_state.process_step == 2:
    result = display_combined_hypothesis_step(hypothesis_collection, call_llm)
    
    if result == "next":
        st.session_state.process_step = 3  # Go to the next sequential step (what was step 4)
        st.rerun()
    elif result == "back":
        st.session_state.process_step = 1
        st.rerun()

# Step 3: File Upload
elif st.session_state.process_step == 3:
    result = display_file_upload_step(db, fs, hypothesis_collection, parse_data)
    
    if result == "next":
        st.session_state.process_step = 4
        st.rerun()
    elif result == "back":
        st.session_state.process_step = 2
        st.rerun()


# Step 4: Interactive Code Review
elif st.session_state.process_step == 4:
    result = display_code_review_step(db, fs, hypothesis_collection)
    
    if result == "next":
        st.session_state.process_step = 5
        st.rerun()
    elif result == "back":
        st.session_state.process_step = 3
        st.rerun()
    elif result == "reload":
        st.rerun()

# Step 5: Evidence Processing
elif st.session_state.process_step == 5:
    result = display_evidence_step(db, fs, hypothesis_collection)
    
    if result == "next":
        st.session_state.process_step = 6
        st.rerun()
    elif result == "back":
        st.session_state.process_step = 4
        st.rerun()
    elif result == "back_to_step4":
        # Special case to go back to step 4 but keep node state
        st.session_state.process_step = 4
        st.rerun()
    elif result == "reload":
        st.rerun()

# Step 6: View Results
elif st.session_state.process_step == 6:
    result = display_results_step(db, fs, hypothesis_collection)
    
    if result == "back":
        st.session_state.process_step = 5
        st.rerun()
    elif result == "home":
        # Reset to start
        st.session_state.process_step = 1
        st.rerun()
