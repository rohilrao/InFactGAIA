import streamlit as st
import sys
import os

# Page configuration
st.set_page_config(page_title="InFactGAIAV3", page_icon="📂", layout="wide")

# Title & Subtitle
st.markdown("### :orange[InFact Demo: Building Trust in Science through Collaborative Evaluation]")
st.markdown("#### :orange[A Gaia Lab Project]")

# 📌 Blockquote
st.markdown("""
    <p class="blockquote">
    "What we should do is create an institution that collects and evaluates scientific evidence and gives out confidence values based on evidence."
    <br>— <a href="https://www.youtube.com/watch?v=zucXnn64qtk&t=314s" target="_blank">Sabine Hossenfelder</a>
    </p>
""", unsafe_allow_html=True)

# Section: Introduction
st.markdown("### :orange[Introduction]")
st.markdown("""
The <b>InFact Project</b> is our attempt to realize this vision. We're building a prototype for a decentralized system 
that evaluates scientific claims and provides a clear measure of confidence based on available evidence.  

Imagine a collaborative platform where scientists and the public can <b>work together</b> to assess the reliability of scientific findings, 
supported by <b>AI-powered analysis</b> and <b>rigorous automated statistics</b>.  
This is the core idea behind InFact.
""", unsafe_allow_html=True)

# Section: The Gaia Network
st.markdown("### :orange[The Gaia Network]")
st.markdown("""
InFact is also envisioned as a <b>demonstration of the capabilities of the</b> 
<a href="https://gaia-lab.de"><b>Gaia Network Protocol</b></a>, the <b>Gaia Lab's main project</b>.  
Visit our website to learn more!
""", unsafe_allow_html=True)

# Make sure we have consistent path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Store the project root in session state for other modules to access
if "project_root" not in st.session_state:
    st.session_state["project_root"] = PROJECT_ROOT

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


# Step 1: AI Model Config
if st.session_state.process_step == 1:
    if display_setup_step():
        st.session_state.process_step = 2
        st.rerun()

#Step 2
# Replace the individual step calls with this:
elif st.session_state.process_step == 2:
    result = display_combined_hypothesis_step(hypothesis_collection)
    
    if result == "next":
        st.session_state.process_step = 3  # Go to the next sequential step (what was step 4)
        st.rerun()
    elif result == "back":
        st.session_state.process_step = 1
        st.rerun()

# Step 3: File Upload
elif st.session_state.process_step == 3:
    result = display_file_upload_step(db, fs, hypothesis_collection)
    
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


# Step 5: View Results
elif st.session_state.process_step == 5:
    result = display_results_step(db, fs, hypothesis_collection)
    
    if result == "back":
        st.session_state.process_step = 4  # Go back to code review
        st.rerun()
    elif result == "add_evidence":  # New case for "Process Another File" button
        st.session_state.process_step = 3  # Go back to file upload/add evidence
        st.rerun()
    elif result == "home":
        # Reset to start
        st.session_state.process_step = 1
        st.rerun()