import streamlit as st
from pymongo import MongoClient
import gridfs
import datetime
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi
import openai
from anthropic import Anthropic
import json
import base64
import pandas as pd
from pathlib import Path
import os
import tempfile
import sys
import time
import streamlit.components.v1 as components
from jinja2 import Template
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds "src" to path
from utils import parse_data 
from AnthropicInFactNode import AnthropicInFactNode
from GptInFactNode import GptInFactNode
from DeepSeekInFactNode import DeepSeekInFactNode
from InFactRenderer import InFactRenderer

# Add the page_utils folder to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "page_utils")))
from _step1_model_config import step_1_model_config
from _step2_enter_hypothesis import enter_hypothesis

process_steps = {
    1: "Setup",
    2: "Create Hypothesis", 
    3: "Generate Summary",
    4: "Upload Files",
    5: "Interactive Code Review",
    6: "Evidence Processing",
    7: "View Results"
}

# Initialize session state if needed
if "process_step" not in st.session_state:
    st.session_state.process_step = 1

# Display step progress
def show_step_progress():
    current_step = st.session_state.process_step
    
    # Create a container for the step progress
    step_container = st.container()
    
    with step_container:
        cols = st.columns(len(process_steps))
        
        for i, (step_num, step_name) in enumerate(process_steps.items()):
            with cols[i]:
                if step_num < current_step:
                    # Completed step
                    st.markdown(f"<div style='text-align: center; color: green;'>✓<br>{step_name}</div>", unsafe_allow_html=True)
                elif step_num == current_step:
                    # Current step
                    st.markdown(f"<div style='text-align: center; font-weight: bold;'>→<br>{step_name}</div>", unsafe_allow_html=True)
                else:
                    # Future step
                    st.markdown(f"<div style='text-align: center; color: gray;'>{step_num}<br>{step_name}</div>", unsafe_allow_html=True)
    
    # Add a separator
    st.markdown("---")

# Call this at the beginning of your app after the title
show_step_progress()

# Helper function for consistent ObjectId handling
def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except:
            return id_value
    return id_value

# ✅ Function to store parsed data in MongoDB under the file entry in `fs.files`
def save_parsed_data_to_file(file_id, parsed_data):
    """
    Updates the MongoDB GridFS file entry with parsed data.
    """
    try:
        file_id = ensure_object_id(file_id)
        db.fs.files.update_one(
            {"_id": file_id},  # Update file by its unique ID
            {"$set": {
                "parsed_data": parsed_data,
                "parsing_complete": True  
            }}
        )
        st.success("✅ Parsed data stored successfully")
    except Exception as e:
        st.error(f"❌ Failed to save parsed data: {str(e)}")

def render_parsed_data(parsed_data, filename):
    """Renders the extracted JSON into an HTML template using Jinja2 and properly displays it in Streamlit."""
    template = Template(TEMPLATE)
    rendered_html = template.render(data=parsed_data, filename=filename)

    # ✅ Ensure Streamlit renders full HTML properly
    components.html(rendered_html, height=600, scrolling=True)

# ------------------------------------------------
# 🔐 MongoDB Connection
# ------------------------------------------------
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["infact_db_v3"]
fs = gridfs.GridFS(db)
hypothesis_collection = db["hypotheses"]

def check_hypothesis_id():
    """Checks if the user-entered ID exists in MongoDB and stores the result."""
    _id = st.session_state.get("hypothesis_id_input", "").strip()
    if not _id:
        st.session_state["id_exists"] = None
        return

    # Look up in DB
    existing = hypothesis_collection.find_one({"_id": _id})
    st.session_state["id_exists"] = True if existing else False

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
    


# ------------------------------------------------
# Helper: Save hypothesis to MongoDB

# ----------------------------
# STEP 1: AI Model Config
# ----------------------------
if st.session_state.process_step == 1:
    step_1_model_config()

# ----------------------------
# STEP 2: Enter Hypothesis
# ----------------------------
elif st.session_state.process_step == 2:
    enter_hypothesis()