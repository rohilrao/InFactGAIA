import streamlit as st
import pymongo
import os
import json
import pandas as pd
import io
import sys
from gridfs import GridFS
from pathlib import Path
from utils import process_evidence, get_latest_html_path
from FetchEvidenceUtils import download_file

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

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

    # Check if the document exists in MongoDB
    document = hypotheses_collection.find_one({"identifier": identifier}, {"_id": 0, "evidence": 1, "hypothesis": 1})

    if not document:
        hypotheses_collection.insert_one({"identifier": identifier, "hypothesis": hypothesis_text, "evidence": []})

    existing_filenames = {fs.get(f_id).filename for f_id in document.get("evidence", []) if fs.exists(f_id)}

    for file_path in evidence_files:
        file_name = file_path.name  # Get only the filename
        if file_name not in existing_filenames:
            if file_path.exists():  # Ensure file is present before opening
                with open(file_path, "rb") as f:
                    file_id = fs.put(f, filename=file_name, content_type="application/octet-stream")
                    hypotheses_collection.update_one({"identifier": identifier}, {"$push": {"evidence": file_id}}, upsert=True)
                print(f"✅ Successfully stored {file_name} from local repo")
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

# ✅ Process Evidence
if st.button("Process Evidence"):
    if not identifier.strip() or not api_key.strip():
        st.error("⚠️ Please enter a Hypothesis Identifier and API Key!")
    else:
        try:
            process_evidence(node_type, identifier, api_key, model, hypothesis)
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
