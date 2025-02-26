import streamlit as st
import pymongo
import os
import json
import pandas as pd
from pathlib import Path
from utils import process_evidence, display_latest_html_result
from FetchEvidenceUtils import download_file

# Initialize MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["infact_db"]
hypotheses_collection = db["hypotheses"]

def save_hypothesis(hypothesis, model, api_key):
    """Save hypothesis and API key to MongoDB."""
    hypotheses_collection.insert_one({
        "hypothesis": hypothesis,
        "model": model,
        "api_key": api_key,
        "evidence": [],
        "results": None
    })

def upload_evidence(hypothesis, file):
    """Save uploaded evidence file."""
    file_path = f"evidence/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    hypotheses_collection.update_one(
        {"hypothesis": hypothesis},
        {"$push": {"evidence": file_path}}
    )
    return file_path

def process_hypothesis(hypothesis, model, api_key):
    """Run the evidence processing logic."""
    base_dir = Path("./hypotheses")
    hypothesis_folder_name = hypothesis.replace(" ", "_")
    results_dir = base_dir / hypothesis_folder_name / "results"
    
    process_evidence(model, hypothesis_folder_name, base_dir, api_key, model, hypothesis)
    
    hypotheses_collection.update_one(
        {"hypothesis": hypothesis},
        {"$set": {"results": str(results_dir)}}
    )
    return results_dir

# Streamlit UI
st.title("InFact - Trust in Science Platform")

# Inputs
api_key = st.text_input("Enter API Key:")
hypothesis = st.text_area("Enter Hypothesis:")
model = st.selectbox("Select AI Model:", ["Anthropic", "GPT", "DeepSeek"])

if st.button("Save Hypothesis"):
    save_hypothesis(hypothesis, model, api_key)
    st.success("Hypothesis saved successfully!")

# Upload Section
st.subheader("Upload Evidence")
uploaded_file = st.file_uploader("Upload document:", type=["pdf", "csv", "png"])
if uploaded_file:
    file_path = upload_evidence(hypothesis, uploaded_file)
    st.success(f"Uploaded {uploaded_file.name}")

# Process Hypothesis
if st.button("Process Evidence"):
    result_path = process_hypothesis(hypothesis, model, api_key)
    st.success("Evidence processed!")
    display_latest_html_result(model, result_path)

# Display Results
st.subheader("Results")
document = hypotheses_collection.find_one({"hypothesis": hypothesis})
if document and document.get("results"):
    st.write("Processed Results:")
    st.write(document["results"])
