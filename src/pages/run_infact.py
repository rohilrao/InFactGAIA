import streamlit as st
import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import utility modules
from page_utils.step_utils import initialize_session_state, show_step_progress
from page_utils.db_utils import initialize_db
from page_utils.setup_step import display_setup_step
from page_utils.hypothesis_step import display_hypothesis_step
from page_utils.summary_step import display_summary_step
from page_utils.file_upload_step import display_file_upload_step

# Import other necessary modules for remaining steps
from anthropic import Anthropic
import openai
import datetime
import json
import tempfile
from pathlib import Path
import pandas as pd
from bson.objectid import ObjectId

# Initialize app
st.title("Hypothesis Wizard")

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

# Step 1: AI Model Config
if st.session_state.process_step == 1:
    if display_setup_step():
        st.session_state.process_step = 2
        st.rerun()

# Step 2: Hypothesis Setup
elif st.session_state.process_step == 2:
    result = display_hypothesis_step(hypothesis_collection)
    
    if result == "next":
        st.session_state.process_step = 3
        st.rerun()
    elif result == "back":
        st.session_state.process_step = 1
        st.rerun()
    elif result == "reload":
        st.rerun()

# Step 3: Hypothesis Refinement & Summary
elif st.session_state.process_step == 3:
    result = display_summary_step(hypothesis_collection, call_llm)
    
    if result == "next":
        st.session_state.process_step = 4
        st.rerun()
    elif result == "back":
        st.session_state.process_step = 2
        st.rerun()

# Step 4: File Upload
elif st.session_state.process_step == 4:
    result = display_file_upload_step(db, fs, hypothesis_collection, parse_data)
    
    if result == "next":
        st.session_state.process_step = 5
        st.rerun()
    elif result == "back":
        st.session_state.process_step = 3
        st.rerun()


# Additional steps follow the same pattern...
# Step 4, 5, 6, 7, etc.

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