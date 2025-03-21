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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds "src" to path

from utils import parse_data 
from jinja2 import Template

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
                "status": "processed"  # Mark file as processed
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
        "GPT": ["gpt-4o"],   # Example GPT models
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

    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
        st.stop()

    # Get the current hypothesis text from DB to ensure we have the latest version
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.session_state["hypothesis_text"] = hypothesis_text  # Keep session state in sync
    
    st.subheader(f"Hypothesis ID: `{hypothesis_id}`")
    st.write(f"Hypothesis: **{hypothesis_text}**")

    # ========== FILE UPLOAD SECTION ========= #
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg", "html", "csv"])

    if uploaded_file:
        file_name = uploaded_file.name
        file_content = uploaded_file.read()

        # ✅ FIXED DUPLICATE CHECK: Check inside `fs.files`
        existing_file = db.fs.files.find_one(
            {"metadata.hypothesis_id": hypothesis_id, "filename": file_name}
        )

        if existing_file:
            st.warning(f"⚠️ A file named **{file_name}** already exists under hypothesis `{hypothesis_id}`. Please upload a different file.")
        else:
            # ✅ Step 2: Save file to GridFS
            file_id = fs.put(
                file_content,
                filename=file_name,
                metadata={
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_text": hypothesis_text
                },
                upload_date=str(datetime.date.today()),
            )

            # ✅ Store latest uploaded file ID in session & mark parsing as active
            st.session_state["latest_uploaded_file_id"] = file_id
            st.session_state["latest_uploaded_filename"] = file_name
            st.session_state["is_parsing"] = True  # 🚀 **NEW: Mark that parsing is happening**
            st.success(f"✅ Uploaded: {file_name}")

            st.rerun()

    # 📂 Fetch & Display existing files
    files = list(db.fs.files.find({"metadata.hypothesis_id": hypothesis_id}))  # ✅ Proper query
    if files:
        st.subheader("Existing Files")

        for file in files:
            file_id = file["_id"]
            filename = file["filename"]
            
            # Check if this file has been processed
            has_parsed_data = "parsed_data" in file
            status_label = "✅ Processed" if has_parsed_data else "⏳ Not processed"

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"📄 **{filename}** ({status_label})")

            with col2:
                with fs.get(file_id) as grid_out:
                    file_content = grid_out.read()
                st.download_button("⬇️ Download", file_content, filename, key=f"download_{file_id}")

            with col3:
                # Show parsed data if it exists
                if has_parsed_data:
                    if st.button("🔍 View Analysis", key=f"view_{file_id}"):
                        st.session_state["viewing_file_id"] = file_id
                        st.session_state["viewing_filename"] = filename
                        st.rerun()

            with col4:
                # ❌ **FIXED: Prevent Deleting While Parsing**
                if st.session_state.get("is_parsing") and st.session_state.get("latest_uploaded_file_id") == file_id:
                    st.button("🔄 Parsing...", disabled=True, key=f"disable_delete_{file_id}")  # **Disable delete**
                else:
                    if st.button("🗑️ Delete", key=f"delete_{file_id}"):
                        fs.delete(ensure_object_id(file_id))
                        st.warning(f"Deleted {filename}")
                        st.rerun()

    # Show the parsed data for a selected file if any
    if "viewing_file_id" in st.session_state:
        view_file_id = ensure_object_id(st.session_state["viewing_file_id"])
        view_filename = st.session_state["viewing_filename"]
        
        # Get file from GridFS
        file_doc = db.fs.files.find_one({"_id": view_file_id})
        
        if file_doc and "parsed_data" in file_doc:
            st.subheader(f"🔍 Analysis for {view_filename}")
            st.write("---")
            render_parsed_data(file_doc["parsed_data"], view_filename)
            
            if st.button("Close Analysis View"):
                del st.session_state["viewing_file_id"]
                del st.session_state["viewing_filename"]
                st.rerun()
            st.write("---")

    # ✅ Step 3: Parse the Latest Uploaded File (ONLY the most recent one)
    if "latest_uploaded_file_id" in st.session_state and st.session_state.get("is_parsing", False):
        file_id = st.session_state["latest_uploaded_file_id"]
        filename = st.session_state["latest_uploaded_filename"]

        # ✅ Reconstruct file from GridFS
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, filename)

        with open(temp_file_path, "wb") as f:
            f.write(fs.get(ensure_object_id(file_id)).read())

        st.write(f"📂 **Temp file created for parsing:** `{temp_file_path}`")

        # ✅ Parse the file
        with st.spinner(f"🔄 Parsing {filename}..."):
            try:
                provider = st.session_state.get("provider")
                model = st.session_state.get("model")
                api_key = st.session_state.get("api_key")

                parsed_data = parse_data(temp_file_path, hypothesis_text, provider, model, api_key)
                
                # Store parsed data in the session temporarily
                st.session_state[f"parsed_data_{file_id}"] = parsed_data
                
                # But also save directly to MongoDB for persistence
                save_parsed_data_to_file(file_id, parsed_data)
                
                # Clean up the temp file
                try:
                    os.remove(temp_file_path)
                except:
                    pass

            except Exception as e:
                st.error(f"❌ Error parsing file {filename}: {str(e)}")
                parsed_data = None

        # ✅ **FINALLY: Mark Parsing as Done**
        st.session_state["is_parsing"] = False

        # ✅ Show parsed data using `render_parsed_data`
        if parsed_data:
            st.subheader(f"🔍 Parsed Data for {filename}")
            render_parsed_data(parsed_data, filename)

    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Step 3"):
            st.session_state.process_step = 3
            st.rerun()
    with col2:
        if st.button("Finish"):
            # Clear temporary state before finishing
            if "latest_uploaded_file_id" in st.session_state:
                del st.session_state["latest_uploaded_file_id"]
                del st.session_state["latest_uploaded_filename"] 
            if "is_parsing" in st.session_state:
                del st.session_state["is_parsing"]
                
            st.success("✅ All steps completed! Your hypothesis and associated files have been saved.")