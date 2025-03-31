import streamlit as st
import sys
import os

# Make sure we have consistent path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Store the project root in session state for other modules to access
if "project_root" not in st.session_state:
    st.session_state["project_root"] = PROJECT_ROOT

# Create directories if they don't exist
node_states_dir = os.path.join(PROJECT_ROOT, "node_states")
os.makedirs(node_states_dir, exist_ok=True)

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
import datetime
import json
import tempfile
from pathlib import Path
import pandas as pd
from bson.objectid import ObjectId

# Add infact to the path
infact_path = os.path.abspath(os.path.join(PROJECT_ROOT, "infact"))
sys.path.append(infact_path)

# Import from utils
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
    if provider == "openai":
        import openai
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.1
        )
        return response.choices[0].message.content

    # For Anthropic
    elif provider == "anthropic":
        from anthropic import Anthropic
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


# Step 6: View Results
elif st.session_state.process_step == 5:
    result = display_results_step(db, fs, hypothesis_collection)
    
    if result == "back":
        st.session_state.process_step = 4
        st.rerun()
    elif result == "home":
        # Reset to start
        st.session_state.process_step = 1
        st.rerun()