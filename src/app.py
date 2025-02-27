import streamlit as st
import pymongo
import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from utils import process_evidence, display_latest_html_result
from FetchEvidenceUtils import download_file
import io
import sys

# Paths and Config
MONGO_DB_PATH = "/home/rrao/software/mongodb/mongodb/bin/mongod"
DB_DATA_PATH = "../mongodb_data"
OPENSSL_PATH = "/home/rrao/software/openssl"

# Set OpenSSL environment variables
os.environ["LD_LIBRARY_PATH"] = f"{OPENSSL_PATH}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["PATH"] = f"{OPENSSL_PATH}/bin:{os.environ.get('PATH', '')}"

# Ensure necessary directories exist
os.makedirs(DB_DATA_PATH, exist_ok=True)
os.makedirs("evidence", exist_ok=True)
os.makedirs("hypotheses", exist_ok=True)

# Connect to MongoDB
try:
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    client.server_info()  # Check if MongoDB is running
except:
    st.warning("Starting MongoDB...")
    subprocess.Popen([MONGO_DB_PATH, "--dbpath", DB_DATA_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st.success("MongoDB started successfully!")

# Initialize MongoDB Collections
db = client["infact_db"]
hypotheses_collection = db["hypotheses"]

def save_hypothesis(hypothesis, model):
    """Save hypothesis to MongoDB (without API key)."""
    if not hypothesis.strip():
        st.error("Hypothesis cannot be empty!")
        return
    
    existing = hypotheses_collection.find_one({"hypothesis": hypothesis})
    if existing:
        st.warning("Hypothesis already exists!")
    else:
        hypotheses_collection.insert_one({
            "hypothesis": hypothesis,
            "model": model,
            "evidence": [],
            "results": None
        })
        st.success("Hypothesis saved successfully!")

def upload_evidence(hypothesis, file):
    """Save uploaded evidence file."""
    if not hypothesis.strip():
        st.error("Please enter a hypothesis before uploading evidence.")
        return
    
    if not hypotheses_collection.find_one({"hypothesis": hypothesis}):
        st.error("Hypothesis not found! Save it first.")
        return

    file_path = os.path.abspath(f"evidence/{file.name}")
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    hypotheses_collection.update_one(
        {"hypothesis": hypothesis},
        {"$push": {"evidence": file_path}}
    )
    st.success(f"Uploaded {file.name}")

def process_hypothesis(hypothesis, node_type, model, api_key):
    """Run the evidence processing logic and display logs in the UI."""
    if not hypothesis.strip():
        st.error("Please enter a hypothesis first.")
        return

    if not api_key.strip():
        st.error("API Key is required to process evidence!")
        return

    base_dir = Path("./hypotheses")
    hypothesis_folder_name = hypothesis.replace(" ", "_")
    results_dir = base_dir / hypothesis_folder_name / "results"
    os.makedirs(results_dir, exist_ok=True)

    # ✅ Capture stdout and stderr
    output_buffer = io.StringIO()
    sys.stdout = output_buffer  # Redirect standard output
    sys.stderr = output_buffer  # Redirect standard error

    try:
        # ✅ Use node_type instead of duplicating model
        process_evidence(node_type, hypothesis_folder_name, base_dir, api_key, model, hypothesis)
        hypotheses_collection.update_one(
            {"hypothesis": hypothesis},
            {"$set": {"results": str(results_dir)}}
        )
        st.success("Evidence processed successfully!")

    except Exception as e:
        st.error(f"Error processing evidence: {e}")

    finally:
        # ✅ Reset stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # ✅ Display logs in Streamlit UI
        process_logs = output_buffer.getvalue()
        if process_logs:
            st.subheader("Processing Logs:")
            st.text_area("Logs:", process_logs, height=300)


# Streamlit UI
st.title("InFact - Trust in Science Platform")

# Inputs
api_key = st.text_input("Enter API Key:", type="password")  # API key stays in memory
hypothesis = st.text_area("Enter Hypothesis:")

# ✅ Step 1: User selects a node type
node_type = st.selectbox("Select Node Type:", ["GPT", "Anthropic", "DeepSeek"])

# ✅ Step 2: Show only relevant models based on node type
model_options = {
    "GPT": ["gpt-4o-2024-08-06", "chatgpt-4o-latest"],
    "Anthropic": ["claude-3-5-sonnet-20241022"],
    "DeepSeek": ["deepseek-chat"]
}

# ✅ Step 3: Model dropdown updates dynamically based on node type selection
model = st.selectbox("Select AI Model:", model_options[node_type])

# ✅ Step 4: Map the node type to the expected format for processing
node_type_mapping = {
    "GPT": "gpt",
    "Anthropic": "anthropic",
    "DeepSeek": "deepseek"
}

node_type = node_type_mapping[node_type]  # Convert UI-friendly name to expected format

if st.button("Save Hypothesis"):
    save_hypothesis(hypothesis, model)

# Upload Section
st.subheader("Upload Evidence")
uploaded_file = st.file_uploader("Upload document:", type=["pdf", "csv", "png"])
if uploaded_file:
    upload_evidence(hypothesis, uploaded_file)

# Process Hypothesis
if st.button("Process Evidence"):
    process_hypothesis(hypothesis, node_type, model, api_key)  # Use API key without storing it

# Display Results
st.subheader("Results")
document = hypotheses_collection.find_one({"hypothesis": hypothesis})
if document and document.get("results"):
    st.write("Processed Results:")
    st.write(document["results"])
