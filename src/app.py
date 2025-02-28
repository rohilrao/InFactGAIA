import streamlit as st
import pymongo
import os
import json
import pandas as pd
import io
import sys
from gridfs import GridFS
from pathlib import Path

from FetchEvidenceUtils import download_file
from gridfs import GridFS
import io
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
import json
from pathlib import Path
import json
import logging
from pathlib import Path
from AnthropicInFactNode import AnthropicInFactNode
from GptInFactNode import GptInFactNode
from DeepSeekInFactNode import DeepSeekInFactNode
from InFactRenderer import InFactRenderer

print("✅ Successfully imported required libraries")

# ✅ Securely Load MongoDB URI from Streamlit Secrets
MONGO_URI = st.secrets["MONGO_URI"]
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

# ✅ Test Connection
try:
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"❌ Connection Error: {e}")

# ✅ Initialize MongoDB Collections
db = client["infact_db"]
hypotheses_collection = db["hypotheses"]
fs = GridFS(db)

# ✅ Ensure the script works when executed from 'src/'
BASE_DIR = Path(__file__).parent.parent  # Move up from 'src/' to project root

# ✅ Function to Preload Evidence from GitHub Repo
def preload_evidence():
    identifier = "example_hypotheses_HumanGHGs_GlobalWarming"
    hypothesis_text = "Human-generated GHG emissions significantly increase global temperatures"

    evidence_files = [
        BASE_DIR / "hypotheses/example_hypotheses_HumanGHGs_GlobalWarming/evidence/greenhouse-gases.html",
        BASE_DIR / "hypotheses/example_hypotheses_HumanGHGs_GlobalWarming/evidence/World_of_Change_Global_Temperatures.html"
    ]

    # 🔹 Check if the document exists
    document = hypotheses_collection.find_one({"identifier": identifier}, {"_id": 0, "evidence": 1, "hypothesis": 1})

    # 🔹 If not found, create a new document with default hypothesis
    if not document:
        print(f"⚠️ Hypothesis '{identifier}' not found. Creating default entry.")
        document = {"evidence": [], "hypothesis": hypothesis_text}
        hypotheses_collection.insert_one({"identifier": identifier, "hypothesis": hypothesis_text, "evidence": []})

    # 🔹 Reload document to ensure it's available
    document = hypotheses_collection.find_one({"identifier": identifier}, {"_id": 0, "evidence": 1, "hypothesis": 1})

    # 🔹 Ensure we only proceed if the document is not None
    if document:
        existing_filenames = {fs.get(f_id).filename for f_id in document.get("evidence", []) if fs.exists(f_id)}
    else:
        print("❌ Error: Unable to retrieve the document after insertion!")
        return  # Exit the function to prevent errors

    # 🔹 Upload files if they don't already exist
    for file_path in evidence_files:
        file_name = file_path.name
        if file_name not in existing_filenames:
            if file_path.exists():
                with open(file_path, "rb") as f:
                    file_id = fs.put(f, filename=file_name, content_type="application/octet-stream")
                    hypotheses_collection.update_one({"identifier": identifier}, {"$push": {"evidence": file_id}}, upsert=True)
                print(f"✅ Successfully stored {file_name}")
            else:
                print(f"❌ File not found: {file_path}")

# ✅ Preload default evidence
preload_evidence()

# ✅ Load existing hypotheses
existing_hypotheses = {
    h["identifier"]: h.get("hypothesis", "No Hypothesis Available")
    for h in hypotheses_collection.find({}, {"_id": 0, "identifier": 1, "hypothesis": 1})
}

# ✅ Streamlit UI
st.title("InFact - Trust in Science Platform")
api_key = st.text_input("Enter API Key:", type="password")

st.subheader("Hypothesis Management")
identifier = st.text_input("Hypothesis Identifier:", "example_hypotheses_HumanGHGs_GlobalWarming")
hypothesis = st.text_area("Enter Hypothesis:", existing_hypotheses.get(identifier, "Human-generated GHG emissions significantly increase global temperatures"))
node_type = st.selectbox("Select Node Type:", ["GPT", "Anthropic", "DeepSeek"])
model_options = {"GPT": ["chatgpt-4o-latest"], "Anthropic": ["claude-3-5-sonnet-20241022"], "DeepSeek": ["deepseek-chat"]}
model = st.selectbox("Select AI Model:", model_options[node_type])
node_type_mapping = {"GPT": "gpt", "Anthropic": "anthropic", "DeepSeek": "deepseek"}
node_type = node_type_mapping[node_type]

# ✅ Ensure identifier is unique
existing_entry = hypotheses_collection.find_one({"$or": [{"identifier": identifier}, {"hypothesis": hypothesis}]})
if existing_entry and existing_entry.get("identifier") != identifier:
    st.error("This hypothesis already exists with a different identifier!")
elif existing_entry and existing_entry.get("hypothesis") != hypothesis:
    st.error("This identifier is already linked to a different hypothesis!")

if st.button("Save Hypothesis"):
    if not identifier.strip():
        st.error("Hypothesis Identifier cannot be empty!")
    elif not hypothesis.strip():
        st.error("Hypothesis cannot be empty!")
    else:
        hypotheses_collection.update_one(
            {"identifier": identifier},
            {"$set": {"hypothesis": hypothesis, "model": model, "evidence": [], "results": None}},
            upsert=True
        )
        st.success("Hypothesis saved successfully!")

# ✅ File Upload
st.subheader("Upload Evidence")
uploaded_file = st.file_uploader("Upload document:", type=["pdf", "csv", "png", "html", "txt"])

if uploaded_file:
    file_data = io.BytesIO(uploaded_file.getbuffer())
    file_id = fs.put(file_data, filename=uploaded_file.name, content_type=uploaded_file.type)
    hypotheses_collection.update_one({"identifier": identifier}, {"$push": {"evidence": file_id}})
    st.success(f"✅ Uploaded {uploaded_file.name} to MongoDB GridFS!")

def get_processed_log_path(node_type, results_dir):
    node_dir = results_dir / node_type  # Node-specific results directory
    return node_dir / f"processed_files_{node_type}.json"  # ✅ Store in the correct folder

def load_processed_files(node_type):
    """Load processed files list from MongoDB instead of local storage."""
    document = db["processed_files"].find_one({"node_type": node_type})
    return set(document["files"]) if document else set()

def save_processed_files(processed_files, node_type):
    """Save processed files list to MongoDB instead of local storage."""
    db["processed_files"].update_one(
        {"node_type": node_type},
        {"$set": {"files": list(processed_files)}},
        upsert=True
    )


def load_or_create_node(node_type, hypothesis, model, api_key):
    """Load or create an InFact node from MongoDB GridFS."""
    
    document = db["node_states"].find_one({"node_type": node_type})

    if document:
        print(f"🔄 Loading existing {node_type} node state from MongoDB GridFS...")
        file_obj = fs.get(document["state_file_id"])
        node_state = json.loads(file_obj.read().decode())  # Read JSON state
    else:
        print(f"✨ Creating new {node_type} InFact node...")
        node_state = {}

    if node_type == "anthropic":
        node = AnthropicInFactNode(hypothesis=hypothesis, api_key=api_key, model=model)
    elif node_type == "gpt":
        node = GptInFactNode(hypothesis=hypothesis, api_key=api_key, model=model)
    elif node_type == "deepseek":
        node = DeepSeekInFactNode(hypothesis=hypothesis, api_key=api_key, model=model)

    return node

def process_evidence(node_type, hypothesis_identifier, base_dir, api_key, model, hypothesis):
    """
    Processes evidence directly from MongoDB GridFS instead of local files.
    """
    db = client["infact_db"]
    hypotheses_collection = db["hypotheses"]
    fs = GridFS(db)

    # Retrieve evidence files from GridFS
    document = hypotheses_collection.find_one({"identifier": hypothesis_identifier})
    if not document or "evidence" not in document or not document["evidence"]:
        print(f"🚫 No evidence files found for {hypothesis_identifier}.")
        return

    print(f"📂 Found {len(document['evidence'])} evidence files in GridFS for processing.")

    # ✅ Fix: Only expect `node`, not three values
    node = load_or_create_node(node_type, hypothesis, model, api_key)

    processed_files = load_processed_files(node_type)

    new_files = []

    for file_id in document["evidence"]:
        try:
            file_obj = fs.get(file_id)
            file_content = file_obj.read()
            file_name = file_obj.filename

            # Process file **directly from memory** instead of disk
            print(f"🔄 Processing file: {file_name} with {node_type}...")
            node.process_data(io.BytesIO(file_content))  # Pass as file-like object
            new_files.append(file_name)

            # ✅ Fix: Save node state to MongoDB instead of a local file
            node_state_json = json.dumps(node.__dict__)  # Convert node state to JSON
            file_id = fs.put(node_state_json.encode(), filename=f"{node_type}_state.json", content_type="application/json")
            db["node_states"].update_one({"node_type": node_type}, {"$set": {"state_file_id": file_id}}, upsert=True)

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

    if new_files:
        print(f"✅ Processed {len(new_files)} new evidence files.")
    else:
        print(f"🚫 No new evidence to process.")

    logging.info(f"✅ Processing completed for {node_type}.")

# ✅ Process Evidence
if st.button("Process Evidence"):
    if not identifier.strip() or not api_key.strip():
        st.error("⚠️ Please enter a Hypothesis Identifier and API Key!")
    else:
        try:
            # Call the modified `process_evidence` that reads from MongoDB
            process_evidence(node_type, identifier, BASE_DIR, api_key, model, hypothesis)

            st.success("📊 Evidence processed successfully! Results stored.")
        except Exception as e:
            st.error(f"Error processing evidence: {e}")

# ✅ Display Results
st.subheader("Results")
document = hypotheses_collection.find_one({"identifier": identifier})

if document and document.get("latest_analysis_result"):
    file_id = document["latest_analysis_result"]
    file_content = fs.get(file_id).read().decode()

    st.write(f"Latest Analysis Results:")
    st.download_button(label="Download Full Analysis", data=file_content, file_name="analysis.html", mime="text/html")
    st.components.v1.html(file_content, height=1200, scrolling=True)
else:
    st.info("No processed analysis results available yet.")


