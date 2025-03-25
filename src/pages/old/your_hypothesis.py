import streamlit as st
from pymongo import MongoClient
import gridfs
import datetime
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi
import openai
from anthropic import Anthropic
import json
import datetime
import base64
import pandas as pd
import streamlit as st
from pathlib import Path
import os
import tempfile
import sys
import time

import streamlit as st
import json
import sys
import io
import os
import tempfile
from pathlib import Path
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi
import datetime
from AnthropicInFactNode import AnthropicInFactNode
from GptInFactNode import GptInFactNode
from DeepSeekInFactNode import DeepSeekInFactNode
from InFactRenderer import InFactRenderer
import sys
import io


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds "src" to path

from utils import parse_data 
from jinja2 import Template


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


# Sample Jinja2 template for displaying the extracted data
TEMPLATE = """
<style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
        margin: 0;
        padding: 0;
    }
    .container {
        width: 90%;
        max-width: 700px;
        margin: 20px auto;
        padding: 20px;
        background: white;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    h3 {
        color: #333;
        text-align: center;
    }
    h4 {
        color: #0056b3;
    }
    .confidence-score {
        font-weight: bold;
    }
    .confidence-high { color: green; }
    .confidence-medium { color: orange; }
    .confidence-low { color: red; }
    pre {
        background-color: #eef;
        padding: 10px;
        border-radius: 5px;
        white-space: pre-wrap;
    }
    .issues, .confidence-box {
        padding: 15px;
        border-radius: 8px;
    }
    .confidence-box { background: #eef5ff; }
    .issues { background: #ffecec; }
</style>

<div class="container">
    <h3>Parsed Data for {{ filename }}</h3>

    {% if data.confidence_assessment %}
        <div class="confidence-box">
            <h4>Confidence Assessment</h4>
            <p><strong>Score:</strong> 
                <span class="confidence-score 
                    {% if data.confidence_assessment.confidence_score >= 0.75 %} confidence-high
                    {% elif data.confidence_assessment.confidence_score >= 0.5 %} confidence-medium
                    {% else %} confidence-low
                    {% endif %}">
                    {{ (data.confidence_assessment.confidence_score * 100)|round(2) }}%
                </span>
            </p>
            <p><strong>Explanation:</strong> {{ data.confidence_assessment.explanation }}</p>

            {% if data.confidence_assessment.key_strengths %}
                <h4>Key Strengths</h4>
                <ul>
                    {% for strength in data.confidence_assessment.key_strengths %}
                    <li>{{ strength }}</li>
                    {% endfor %}
                </ul>
            {% endif %}

            {% if data.confidence_assessment.key_limitations %}
                <h4>Key Limitations</h4>
                <ul>
                    {% for limitation in data.confidence_assessment.key_limitations %}
                    <li>{{ limitation }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        </div>
    {% endif %}

    {% if data.numerical_values %}
        <h4>Numerical Values</h4>
        <pre>{{ data.numerical_values | join(", ") }}</pre>
    {% endif %}

    {% if data.metadata %}
        <h4>Metadata</h4>
        <pre>{{ data.metadata | tojson(indent=2) }}</pre>
    {% endif %}

    {% if data.issues %}
        <div class="issues">
            <h4>Issues</h4>
            <ul>
                {% for issue in data.issues %}
                <li>{{ issue }}</li>
                {% endfor %}
            </ul>
        </div>
    {% endif %}
</div>
"""
import streamlit.components.v1 as components

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
# Wizard Setup
# ------------------------------------------------
if "process_step" not in st.session_state:
    st.session_state.process_step = 1

#st.title("Hypothesis Wizard")

# ----------------------------
# STEP 1: AI Model Config
# ----------------------------
if st.session_state.process_step == 1:
    st.header("Step 1: AI Model Configuration")
    
    provider = st.selectbox("Select Provider:", ["GPT", "Anthropic"])  # Only GPT or Anthropic
    model_options = {
        "GPT": ["gpt-4o", "chatgpt-4o-latest"],   # Example GPT models
        "Anthropic": ["claude-3-5-sonnet"]   # Example Anthropic model
    }
    model = st.selectbox("Select Model:", model_options[provider])
    api_key = st.text_input("Enter API Key:", type="password")

    if st.button("Next →"):
        if not provider or not model or not api_key.strip():
            st.warning("Please fill in all fields before proceeding.")
        else:
            st.session_state["provider"] = provider
            st.session_state["model"] = model
            st.session_state["api_key"] = api_key
            st.session_state.process_step = 2
            st.rerun()

# ----------------------------
# STEP 2: Hypothesis Setup
# ----------------------------

elif st.session_state.process_step == 2:
    st.header("Step 2: Hypothesis Setup")

    st.markdown("""
    ### Enter a short identifier (ID) for the hypothesis.
    - If the ID already exists, we will load it in read-only form.
    - If the ID does not exist, you can create a new hypothesis below.
    """)

    # Ensure we track existence checks
    if "id_exists" not in st.session_state:
        st.session_state["id_exists"] = None

    # 1) ID input with on_change to dynamically check the DB
    st.text_input(
        "Hypothesis ID (short name)",
        key="hypothesis_id_input",
        on_change=check_hypothesis_id  # your existing function; sets st.session_state["id_exists"] to True/False
    )

    # 2) Check existence
    hypothesis_id = st.session_state.get("hypothesis_id_input", "").strip()
    loaded_text = None

    if hypothesis_id and st.session_state["id_exists"] is True:
        # The ID exists in DB => fetch its text
        doc = hypothesis_collection.find_one({"_id": hypothesis_id})
        if doc:
            loaded_text = doc["text"]
        else:
            # If 'id_exists' = True but doc not found => inconsistent, but let's handle gracefully
            st.error("Inconsistent state: ID said to exist, but not found in DB.")
            st.stop()

        # Put the existing text in session_state for display (read-only)
        st.session_state["hypothesis_text_input"] = loaded_text

        st.info(f"Loaded existing hypothesis with ID '{hypothesis_id}'. You cannot overwrite it.")
    elif hypothesis_id and st.session_state["id_exists"] is False:
        st.warning("No existing hypothesis found for this ID. You can create a new one below.")
    else:
        st.write("Please enter an ID above to check availability.")

    # 3) Display text area
    #    If ID exists => show read-only text area with loaded_text
    #    If ID does not exist => editable text area for new text
    if st.session_state["id_exists"] is True and loaded_text:
        # Show read-only text area for existing ID
        st.markdown("#### Existing Hypothesis Text (read-only):")
        st.text_area(
            "Existing Hypothesis Text",
            value=loaded_text,
            disabled=True
        )
    else:
        # ID does not exist, so let user type a new text
        st.markdown("#### New Hypothesis Text (cannot be edited after creation):")
        st.text_area(
            "Enter your hypothesis details here",
            key="hypothesis_text_input"  # updatable in session
        )

    # 4) Navigation Buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # "← Back"
        if st.button("← Back"):
            st.session_state.process_step = 1
            st.rerun()

    with col2:
        # Only show "Create Hypothesis" if ID doesn't exist
        if st.session_state["id_exists"] is False and hypothesis_id:
            if st.button("Create Hypothesis"):
                new_text = st.session_state.get("hypothesis_text_input", "").strip()
                if not new_text:
                    st.warning("⚠️ Please enter text before creating a new hypothesis.")
                    st.stop()

                # Insert new doc
                hypothesis_collection.insert_one({
                    "_id": hypothesis_id,
                    "original_text": new_text,
                    "text": new_text,
                    "auto_summary": None
                })

                # Mark as created
                st.session_state["hypothesis_id"] = hypothesis_id

                # 🔥 FORCE CHECK so button disappears on rerun
                check_hypothesis_id()

                st.success(f"✅ Created new hypothesis with ID '{hypothesis_id}'")
                time.sleep(1)
                st.rerun()
        else:
            # If ID exists, user can't create or overwrite
            st.caption("No creation needed if ID already exists.")

    with col3:
        # "Next →" button
        # We only allow going forward if we have an existing or newly created hypothesis_id
        # i.e. either st.session_state["id_exists"] is True OR we've just created a new one
        already_in_db = (st.session_state["id_exists"] is True and hypothesis_id)
        newly_created = st.session_state.get("hypothesis_id") == hypothesis_id and hypothesis_id
        can_proceed = already_in_db or newly_created

        if can_proceed:
            if st.button("Next →"):
                # Use a single set operation for the session state
                st.session_state["hypothesis_id"] = hypothesis_id
                
                # Also store the hypothesis text for later use
                hypothesis_doc = hypothesis_collection.find_one({"_id": hypothesis_id})
                if hypothesis_doc:
                    st.session_state["hypothesis_text"] = hypothesis_doc["text"]
                
                st.session_state.process_step = 3
                st.rerun()

# ----------------------------
# STEP 3: Generate Summary
# ----------------------------
elif st.session_state.process_step == 3:
    st.header("Step 3: Hypothesis Refinement & Summary")

    hypothesis_id = st.session_state.get("hypothesis_id")
    if not hypothesis_id:
        st.warning("No Hypothesis ID found. Please go back to Step 2.")
        st.stop()

    # Retrieve hypothesis from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.warning("Hypothesis not found in DB. Please go back and create one.")
        st.stop()

    # Display the current hypothesis in minimal form
    st.write("#### Current Hypothesis:")
    st.write(f"**{hypothesis_entry['text']}**")

    # Keep original text on hand for rewriting (if not already set)
    original_text = hypothesis_entry.get("original_text", hypothesis_entry["text"])

    # ─────────────────────────────────────────────────────────────────
    # Automatically Reformulate as Yes/No with Progress Indicator
    # ─────────────────────────────────────────────────────────────────
    need_reformulation = False
    if "original_text" not in hypothesis_entry:
        need_reformulation = True
    elif hypothesis_entry["original_text"] == hypothesis_entry["text"]:
        need_reformulation = True

    if need_reformulation:
        with st.spinner("🔄 Reformulating hypothesis..."):
            prompt_reformulate = (
                f"Given this hypothesis:\n\n'{original_text}'\n\n"
                "Rewrite/Reformulate it as a clear, concise Yes-No question. " 
                "The answer to the reformulated question should be either 'Yes' or 'No'. "
                "Keep your response brief — ONLY return the reformulated question, nothing else."
            )
            
            # Call LLM
            yes_no_formulation = call_llm(
                provider=st.session_state["provider"],
                model=st.session_state["model"],
                api_key=st.session_state["api_key"],
                prompt_text=prompt_reformulate
            ).strip()
            
            # Validate the response (basic check that it ends with a question mark)
            if not yes_no_formulation.endswith('?'):
                yes_no_formulation = yes_no_formulation.rstrip('.') + '?'
            
            # Update DB
            hypothesis_collection.update_one(
                {"_id": hypothesis_id},
                {"$set": {
                    "original_text": original_text,  # store original if not set
                    "text": yes_no_formulation       # store newly refined text
                }}
            )
            
            # Refresh the data in session state for use in Step 4
            st.session_state["hypothesis_text"] = yes_no_formulation
            
            # Refresh the data in current view
            hypothesis_entry["original_text"] = original_text
            hypothesis_entry["text"] = yes_no_formulation
    
    st.write("#### Refined Hypothesis:")
    st.write(f"**{hypothesis_entry['text']}**")

    # ─────────────────────────────────────────────────────────────────
    # Automatically Generate Summary with Stepwise Progress Indication
    # ─────────────────────────────────────────────────────────────────
    if "auto_summary" not in hypothesis_entry or hypothesis_entry["auto_summary"] is None:
        with st.spinner("🔄 Identifying current state of the field..."):
            prompt_summary = (
                f"Given this yes/no hypothesis question:\n\n'{hypothesis_entry['text']}'\n\n"
                "Please provide:\n"
                "1. **Summarize existing knowledge** (short crisp points, with sources and links if possible).\n"
                "2. **Highlight key controversies** (if any).\n"
                "3. **Discuss relevant data** under two headings:\n"
                "   - ✅ Evidence in Favor\n"
                "   - ❌ Potentially Refuting Evidence\n\n"
                "Keep it succinct, well-structured, and visually clear.\n\n"
                "**Expected Output Format:**\n\n"
                "### Existing Knowledge:\n"
                "- 🔹 [Key fact 1] (Source)\n"
                "- 🔹 [Key fact 2] (Source)\n"
                "- 🔹 [Key fact 3] (Source)\n\n"
                "### Controversies:\n"
                "- ❗ [Main controversy 1]\n"
                "- ❗ [Main controversy 2]\n\n"
                "### Relevant Data:\n\n"
                "#### ✅ Evidence in Favor:\n"
                "- [Supporting evidence 1] (Study/Source)\n"
                "- [Supporting evidence 2] (Study/Source)\n\n"
                "#### ❌ Potentially Refuting Evidence:\n"
                "- [Counter evidence 1] (Study/Source)\n"
                "- [Counter evidence 2] (Study/Source)\n"
            )
            
            with st.spinner("🔄 Identifying key controversies..."):
                llm_response = call_llm(
                    provider=st.session_state["provider"],
                    model=st.session_state["model"],
                    api_key=st.session_state["api_key"],
                    prompt_text=prompt_summary
                )
                
            with st.spinner("🔄 Identifying supporting and refuting evidence..."):
                hypothesis_collection.update_one(
                    {"_id": hypothesis_id},
                    {"$set": {"auto_summary": llm_response}}
                )
                
                # Refresh the data
                hypothesis_entry["auto_summary"] = llm_response
    
    st.write("#### Summary:")
    st.write(hypothesis_entry["auto_summary"])

    # ─────────────────────────────────────────────────────────────────
    # Navigation at the Bottom
    # ─────────────────────────────────────────────────────────────────
    st.write("---")  # Just a horizontal rule to separate content from nav
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("← Back to Step 2"):
            st.session_state.process_step = 2
            st.rerun()

    with col2:
        if st.button("Next → to File Upload"):
            st.session_state.process_step = 4
            st.rerun()



# ----------------------------
# STEP 4: File Manager
# ----------------------------
elif st.session_state.process_step == 4:
    st.header("Step 4: Upload Files for Your Hypothesis")

    # Get hypothesis information
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
        st.stop()

    # Get the current hypothesis text from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.session_state["hypothesis_text"] = hypothesis_text
    
    # Simple hypothesis display
    st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
    st.write(f"**Hypothesis:** {hypothesis_text}")
    st.markdown("---")
    
    # Print current files for debugging
    existing_files = list(db.fs.files.find({"metadata.hypothesis_id": hypothesis_id}))
    existing_filenames = [file["filename"] for file in existing_files]
    print(f"DEBUG - Current files for hypothesis ID '{hypothesis_id}': {existing_filenames}")
    
    # Simple file uploader
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg", "html", "csv"])
    
    # Track if we're currently parsing
    is_parsing = st.session_state.get("is_parsing", False)
    
    # Logic for handling file upload
    if uploaded_file and not is_parsing:
        file_name = uploaded_file.name
        print(f"DEBUG - User selected file: '{file_name}'")
        
        # Check if this file already exists for THIS hypothesis
        is_duplicate = file_name in existing_filenames
        
        if is_duplicate:
            print(f"DEBUG - DUPLICATE DETECTED: File '{file_name}' already exists for hypothesis ID '{hypothesis_id}'")
            print(f"DEBUG - Existing files for this hypothesis: {existing_filenames}")
            st.write(f"⚠️ A file named '{file_name}' already exists for this hypothesis. Please choose a different file.")
        else:
            print(f"DEBUG - File '{file_name}' is not a duplicate for hypothesis ID '{hypothesis_id}'")
            # Show upload button if not a duplicate
            if st.button("Upload File", key="upload_button"):
                print(f"DEBUG - Uploading file '{file_name}' for hypothesis ID '{hypothesis_id}'")
                # Read file content
                file_content = uploaded_file.read()
                
                # Save to GridFS
                file_id = fs.put(
                    file_content,
                    filename=file_name,
                    metadata={
                        "hypothesis_id": hypothesis_id,
                        "hypothesis_text": hypothesis_text
                    },
                    status="unprocessed",
                    upload_date=str(datetime.date.today()),
                )
                
                print(f"DEBUG - File successfully uploaded with ID: {file_id}")
                
                # Store file ID in session
                st.session_state["current_file_id"] = file_id
                st.session_state["current_filename"] = file_name
                st.session_state["is_parsing"] = True
                
                # Rerun to reflect state changes
                st.rerun()
    
    # Process file if needed
    if is_parsing and "current_file_id" in st.session_state:
        file_id = st.session_state["current_file_id"]
        filename = st.session_state["current_filename"]
        print(f"DEBUG - Beginning to process file '{filename}' with ID {file_id}")
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, filename)
        
        # Get file content
        with open(temp_file_path, "wb") as f:
            f.write(fs.get(ensure_object_id(file_id)).read())
        
        print(f"DEBUG - Created temporary file at '{temp_file_path}'")
        
        # Process the file
        try:
            provider = st.session_state.get("provider")
            model = st.session_state.get("model")
            api_key = st.session_state.get("api_key")
            
            print(f"DEBUG - Parsing file '{filename}' with provider '{provider}' and model '{model}'")
            parsed_data = parse_data(temp_file_path, hypothesis_text, provider, model, api_key)
            
            # Save parsed data
            print(f"DEBUG - Saving parsed data for file '{filename}'")
            save_parsed_data_to_file(file_id, parsed_data)
            
            # Clean up
            try:
                os.remove(temp_file_path)
                print(f"DEBUG - Removed temporary file '{temp_file_path}'")
            except Exception as e:
                print(f"DEBUG - Failed to remove temp file: {str(e)}")
            
            # Mark parsing as complete
            st.session_state["is_parsing"] = False
            print(f"DEBUG - Finished processing file '{filename}'")
            
            # Display parsed data
            render_parsed_data(parsed_data, filename)
            
        except Exception as e:
            print(f"DEBUG - ERROR processing file '{filename}': {str(e)}")
            st.session_state["is_parsing"] = False
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Step 3"):
            # Clean up session state
            for key in ["is_parsing", "current_file_id", "current_filename"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.session_state.process_step = 3
            st.rerun()
            
    with col2:
        if st.button("Next → to Code Review"):
            # Clean up any temporary processing state
            for key in ["is_parsing", "current_file_id", "current_filename"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            # Advance to the next step
            st.session_state.process_step = 5
            st.rerun()
# ----------------------------
# STEP 5: Interactive Code Review
# ----------------------------
elif st.session_state.process_step == 5:
    st.header("Step 5: Interactive Code Review")
    
    # Get hypothesis information
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
        st.stop()

    # Get the current hypothesis text and file info from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
    st.write(f"**Hypothesis:** {hypothesis_text}")
    st.markdown("---")
    
    # Get file ID if it exists in session state
    file_id = st.session_state.get("current_file_id", None)
    
    if not file_id:
        # Let user select an unprocessed file if not already selected
        unprocessed_files = list(db.fs.files.find({"metadata.hypothesis_id": hypothesis_id, "status": "unprocessed"}))
        
        if not unprocessed_files:
            st.warning("No unprocessed files found. Please upload files in Step 4.")
            
            if st.button("← Back to Step 4"):
                st.session_state.process_step = 4
                st.rerun()
            st.stop()
        
        st.subheader("Select a file to analyze")
        file_options = {file["filename"]: file["_id"] for file in unprocessed_files}
        selected_filename = st.selectbox("Choose file:", list(file_options.keys()))
        
        if st.button("Select File"):
            st.session_state["current_file_id"] = file_options[selected_filename]
            st.session_state["current_filename"] = selected_filename
            st.rerun()
    
    else:
        # We have a file to analyze
        filename = st.session_state.get("current_filename", "Selected File")
        st.subheader(f"Analyzing: {filename}")
        
        # Check if we already have parsed data and node
        parsed_data = st.session_state.get("parsed_data", None)
        node = st.session_state.get("node", None)
        
        if not parsed_data or not node:
            # Fetch file content and create temporary file
            file_obj = fs.get(ensure_object_id(file_id))
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, filename)
            
            with open(temp_file_path, "wb") as f:
                f.write(file_obj.read())
                
            # Initialize node and process file to extract data
            with st.spinner("Initializing and parsing file data..."):
                provider = st.session_state.get("provider", "anthropic")
                model = st.session_state.get("model", "claude-3-5-sonnet-20241022")
                api_key = st.session_state.get("api_key", "")
                
                try:
                    # Create appropriate node type
                    if provider.lower() == "anthropic":
                        node = AnthropicInFactNode(
                            hypothesis=hypothesis_text,
                            api_key=api_key,
                            model=model
                        )
                    elif provider.lower() == "gpt":
                        node = GptInFactNode(
                            hypothesis=hypothesis_text,
                            api_key=api_key,
                            model=model
                        )
                    elif provider.lower() == "deepseek":
                        node = DeepSeekInFactNode(
                            hypothesis=hypothesis_text,
                            api_key=api_key,
                            model=model
                        )
                    else:
                        st.error(f"Unknown provider: {provider}")
                        st.stop()
                    
                    # Process data interactively to get parsed data
                    parsed_data, metadata = node.process_data_interactively(temp_file_path)
                    
                    # Store in session state
                    st.session_state["parsed_data"] = parsed_data
                    st.session_state["metadata"] = metadata
                    st.session_state["node"] = node
                    st.success("File parsed successfully!")
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        st.warning(f"Failed to remove temporary file: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error parsing file: {str(e)}")
                    st.stop()
        
        # Display parsed data in collapsible section
        with st.expander("Parsed Data (Click to expand)"):
            st.json(parsed_data)
        
        # Check if we have generated code
        generated_code = st.session_state.get("generated_code", None)
        
        if not generated_code:
            if st.button("Generate Analysis Code"):
                with st.spinner("Generating analysis code..."):
                    try:
                        # Get the node from session state
                        node = st.session_state.get("node")
                        if not node:
                            st.error("Session expired. Please start over.")
                            st.stop()
                        
                        # Generate code using our interactive function
                        code = node.interactive_analyze_data(parsed_data)
                        st.session_state["generated_code"] = code
                        st.success("Code generated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating code: {str(e)}")
        
        else:
            # Display the code in an editable text area
            st.subheader("Review and Edit Analysis Code")
            edited_code = st.text_area("Analysis Code", value=generated_code, height=400)
            
            # Check if code has been modified
            if edited_code != generated_code:
                st.session_state["generated_code"] = edited_code
                st.info("Code has been modified. Please verify it before executing.")
            
            # Allow user to provide feedback and regenerate code
            feedback_col1, feedback_col2 = st.columns([3, 1])
            
            with feedback_col1:
                feedback = st.text_area("Feedback for code improvement (optional)", 
                                         placeholder="Provide feedback on what to improve...")
            
            with feedback_col2:
                if st.button("Regenerate Code"):
                    if feedback:
                        with st.spinner("Regenerating code based on feedback..."):
                            try:
                                # Get the node
                                node = st.session_state.get("node")
                                if not node:
                                    st.error("Session expired. Please start over.")
                                    st.stop()
                                
                                # Generate new code with feedback
                                feedback_prompt = f"""
                                Here is the original code:
                                
                                ```python
                                {edited_code}
                                ```
                                
                                User feedback:
                                {feedback}
                                
                                Please improve the code based on this feedback. The code should still:
                                1. Be a function named `calculate_log_likelihoods`
                                2. Take a single dict parameter and return a tuple of (l_plus, l_minus)
                                3. Calculate log likelihoods for the hypothesis: "{hypothesis_text}"
                                4. Be ready to execute as-is
                                
                                Return only the improved Python code.
                                """
                                
                                message = node.client.messages.create(
                                    model=node.model,
                                    max_tokens=8192,
                                    temperature=0.1,
                                    messages=[{
                                        "role": "user",
                                        "content": feedback_prompt
                                    }]
                                )
                                
                                response_text = node._get_message_text(message)
                                
                                # Extract code using autogen
                                from autogen.code_utils import extract_code
                                extracted_code = extract_code(response_text)
                                
                                if not extracted_code:
                                    st.error("No code block found in AI response")
                                    st.stop()
                                
                                # Get the first Python code block
                                improved_code = None
                                for lang, code_block in extracted_code:
                                    if lang.lower() in ['python', 'py', '']:
                                        improved_code = code_block
                                        break
                                
                                if not improved_code:
                                    st.error("No Python code block found in AI response")
                                    st.stop()
                                
                                st.session_state["generated_code"] = improved_code
                                st.success("Code regenerated successfully!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error regenerating code: {str(e)}")
                    else:
                        st.warning("Please provide feedback to guide code regeneration.")
            
            # Execute code button
            st.subheader("Validate Code")
            execute_col1, execute_col2 = st.columns([1, 1])
            
            with execute_col1:
                if st.button("Validate and Test Code"):
                    with st.spinner("Testing code execution..."):
                        try:
                            # Get the node
                            node = st.session_state.get("node")
                            if not node:
                                st.error("Session expired. Please start over.")
                                st.stop()
                            
                            # Execute code
                            l_plus, l_minus = node.execute_analysis_code(edited_code, parsed_data)
                            
                            st.session_state["l_plus"] = l_plus
                            st.session_state["l_minus"] = l_minus
                            st.session_state["validated_code"] = edited_code
                            
                            # Convert log odds to probability for display
                            p_h_given_d = 1 / (1 + math.exp(-l_plus + l_minus))
                            
                            st.success("Code executed successfully!")
                            st.write(f"**l_plus (log P(data | hypothesis)):** {l_plus:.4f}")
                            st.write(f"**l_minus (log P(data | not hypothesis)):** {l_minus:.4f}")
                            st.write(f"**Probability of hypothesis given this data:** {p_h_given_d:.2%}")
                            
                        except Exception as e:
                            st.error(f"Code execution failed: {str(e)}")
                            st.info("Please revise the code and try again.")
            
            # If code has been validated, allow proceeding to next step
            if "validated_code" in st.session_state:
                with execute_col2:
                    if st.button("Continue to Processing"):
                        # Store necessary information in session state
                        st.session_state["file_ready_for_processing"] = True
                        st.session_state.process_step = 6
                        st.rerun()
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Step 4"):
            # Clean up session state
            for key in ["current_file_id", "current_filename", "parsed_data", 
                       "generated_code", "node", "validated_code", "l_plus", "l_minus"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.session_state.process_step = 4
            st.rerun()

# ----------------------------
# STEP 6: Evidence Processing
# ----------------------------
elif st.session_state.process_step == 6:
    st.header("Step 6: Evidence Processing")
    
    # Get hypothesis information
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
        st.stop()

    # Get the current hypothesis text and file info from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
    st.write(f"**Hypothesis:** {hypothesis_text}")
    st.markdown("---")
    
    # Check if we have a file ready for processing
    if not st.session_state.get("file_ready_for_processing", False):
        st.warning("No file ready for processing. Please complete Step 5 first.")
        
        if st.button("← Back to Step 5"):
            st.session_state.process_step = 5
            st.rerun()
        st.stop()
    
    # Get essential data from session state
    file_id = st.session_state.get("current_file_id")
    filename = st.session_state.get("current_filename")
    parsed_data = st.session_state.get("parsed_data")
    validated_code = st.session_state.get("validated_code")
    l_plus = st.session_state.get("l_plus")
    l_minus = st.session_state.get("l_minus")
    node = st.session_state.get("node")
    
    if not node or not validated_code:
        st.error("Required session data is missing. Please go back to Step 5.")
        if st.button("← Return to Step 5"):
            st.session_state.process_step = 5
            st.rerun()
        st.stop()
    
    st.subheader(f"Processing file: {filename}")
    
    # Show a summary of the analysis so far
    st.write("### Analysis Summary")
    
    # Convert log odds to probability for display
    p_h_given_d = 1 / (1 + math.exp(-l_plus + l_minus))
    
    st.write(f"**l_plus (log P(data | hypothesis)):** {l_plus:.4f}")
    st.write(f"**l_minus (log P(data | not hypothesis)):** {l_minus:.4f}")
    st.write(f"**Probability of hypothesis given this data:** {p_h_given_d:.2%}")
    
    # Check if we've already processed the file
    if st.session_state.get("file_processed", False):
        st.success("File processing complete!")
        st.write("### Final Node State and Analysis")
        
        # Display node state visualization
        if node:
            # Get current posterior and convert to probability
            posterior_prob = 1 / (1 + math.exp(-node.current_posterior))
            
            # Display current probability
            st.write(f"**Current probability of hypothesis:** {posterior_prob:.2%}")
            
            # Display confidence interval if available
            lower, upper = node._calculate_uncertainty()
            st.write(f"**95% Confidence Interval:** ({lower:.2%}, {upper:.2%})")
            
            # Create a progress bar to visualize probability
            st.progress(posterior_prob)
            
            # Display data points table
            if node.data_points:
                st.write("### Evidence Summary")
                
                # Create a DataFrame for display
                data_rows = []
                for i, dp in enumerate(node.data_points):
                    metadata = dp.get('metadata', {})
                    filename = metadata.get('filename', f"Data point {i+1}")
                    
                    # Calculate Bayes factor
                    bayes_factor = math.exp(dp['l_plus'] - dp['l_minus'])
                    
                    # Map strength of evidence
                    if bayes_factor < 1:
                        evidence_strength = "Evidence against hypothesis"
                        evidence_color = "red"
                    elif bayes_factor < 3:
                        evidence_strength = "Weak evidence"
                        evidence_color = "orange"
                    elif bayes_factor < 10:
                        evidence_strength = "Moderate evidence"
                        evidence_color = "blue"
                    elif bayes_factor < 30:
                        evidence_strength = "Strong evidence"
                        evidence_color = "green"
                    else:
                        evidence_strength = "Very strong evidence"
                        evidence_color = "darkgreen"
                    
                    data_rows.append({
                        "File": filename,
                        "l_plus": round(dp['l_plus'], 4),
                        "l_minus": round(dp['l_minus'], 4),
                        "Bayes Factor": round(bayes_factor, 2),
                        "Evidence Strength": evidence_strength
                    })
                
                import pandas as pd
                df = pd.DataFrame(data_rows)
                st.dataframe(df)
        
        # Option to download node state
        if "node_state_file_id" in st.session_state:
            node_state_file_id = st.session_state["node_state_file_id"]
            if fs.exists(node_state_file_id):
                st.download_button(
                    label="⬇️ Download Node State",
                    data=fs.get(node_state_file_id).read(),
                    file_name=f"{filename}_node_state.json",
                    mime="application/json"
                )
        
        # Option to download analysis
        if "analysis_file_id" in st.session_state:
            analysis_file_id = st.session_state["analysis_file_id"]
            if fs.exists(analysis_file_id):
                file_content = fs.get(analysis_file_id).read().decode()
                st.download_button(
                    label="⬇️ Download Analysis Report",
                    data=file_content,
                    file_name=f"{filename}_analysis.html",
                    mime="text/html"
                )
                st.components.v1.html(file_content, height=600, scrolling=True)
    
    else:
        # Display the validated code in a read-only text area
        with st.expander("Review Finalized Analysis Code"):
            st.code(validated_code, language="python")
        
        # Offer to process the file
        st.subheader("Process File")
        st.write("Click below to process the file using the validated analysis code.")
        
        process_col1, process_col2 = st.columns([1, 1])
        
        with process_col1:
            if st.button("🚀 Process Evidence File"):
                log_placeholder = st.empty()  # Create placeholder for logs
                output_buffer = StreamToLogger(log_placeholder)
                sys.stdout, sys.stderr = output_buffer, output_buffer  # Redirect logs to UI
                
                try:
                    with st.spinner("Processing file..."):
                        # Fetch file content and create temporary file
                        file_obj = fs.get(ensure_object_id(file_id))
                        temp_dir = tempfile.gettempdir()
                        temp_file_path = os.path.join(temp_dir, filename)
                        
                        with open(temp_file_path, "wb") as f:
                            f.write(file_obj.read())
                        
                        # Process the data interactively with validated code
                        new_posterior, (lower, upper) = node.process_data_interactively(temp_file_path, validated_code)
                        
                        # Save node state to a temporary file
                        temp_node_state_path = os.path.join(temp_dir, f"node_state_{filename}.json")
                        node.save(temp_node_state_path)
                        
                        # Upload node state to GridFS
                        with open(temp_node_state_path, "r", encoding="utf-8") as f:
                            node_state_content = f.read()
                        
                        state_file_id = fs.put(
                            node_state_content.encode(),
                            filename=f"node_state_{filename}.json",
                            content_type="application/json"
                        )
                        
                        # Render analysis
                        temp_analysis_path = os.path.join(temp_dir, f"analysis_{filename}.html")
                        renderer = InFactRenderer()
                        renderer.render_analysis(node, temp_analysis_path)
                        
                        # Upload analysis to GridFS
                        with open(temp_analysis_path, "r", encoding="utf-8") as f:
                            analysis_content = f.read()
                        
                        analysis_file_id = fs.put(
                            analysis_content.encode(),
                            filename=f"analysis_{filename}.html",
                            content_type="text/html"
                        )
                        
                        # Update file status in database
                        db.fs.files.update_one(
                            {"_id": ensure_object_id(file_id)},
                            {
                                "$set": {
                                    "status": "processed",
                                    "node_state_file_id": state_file_id,
                                    "last_processed_at": datetime.datetime.utcnow(),
                                    "analysis_file_id": analysis_file_id,
                                }
                            }
                        )
                        
                        # Store IDs in session state
                        st.session_state["node_state_file_id"] = state_file_id
                        st.session_state["analysis_file_id"] = analysis_file_id
                        st.session_state["file_processed"] = True
                        
                        # Cleanup temporary files
                        try:
                            os.remove(temp_file_path)
                            os.remove(temp_node_state_path)
                            os.remove(temp_analysis_path)
                        except Exception as e:
                            st.warning(f"Failed to remove temporary files: {str(e)}")
                        
                        st.success("File processed successfully!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                
                finally:
                    # Restore stdout/stderr
                    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        
        with process_col2:
            if st.button("Edit Analysis Code"):
                # Go back to code editing in step 5
                st.session_state["file_ready_for_processing"] = False
                st.session_state.process_step = 5
                st.rerun()
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back to Step 5"):
            # Keep most session state but reset processing flag
            st.session_state["file_ready_for_processing"] = False
            st.session_state.process_step = 5
            st.rerun()
    
    with col2:
        if st.button("Process New Evidence"):
            # Clean up session state for new file
            for key in ["current_file_id", "current_filename", "parsed_data", 
                       "generated_code", "validated_code", "l_plus", "l_minus",
                       "file_ready_for_processing", "file_processed",
                       "node_state_file_id", "analysis_file_id"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Keep the node object to maintain state
            st.session_state.process_step = 5
            st.rerun()
    
    with col3:
        if st.button("View All Results"):
            # Navigate to a hypothetical results page
            st.session_state.process_step = 7  # Assuming step 7 is results view
            st.rerun()