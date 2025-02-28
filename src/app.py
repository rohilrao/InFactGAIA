import streamlit as st
import pymongo
import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from utils import process_evidence, display_latest_html_result, get_latest_html_path
from FetchEvidenceUtils import download_file
import io
import sys
from gridfs import GridFS
import tempfile
import threading

# Paths and Config
MONGO_DB_PATH = "/home/rrao/software/mongodb/mongodb/bin/mongod"
DB_DATA_PATH = "/home/rrao/projects/InFactGAIA/mongodb_data"
OPENSSL_PATH = "/home/rrao/software/openssl"

# Set OpenSSL environment variables
os.environ["LD_LIBRARY_PATH"] = f"{OPENSSL_PATH}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["PATH"] = f"{OPENSSL_PATH}/bin:{os.environ.get('PATH', '')}"

# Ensure necessary directories exist
os.makedirs(DB_DATA_PATH, exist_ok=True)
os.makedirs("hypotheses", exist_ok=True)

# Connect to MongoDB
try:
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    client.server_info()
except:
    st.warning("Starting MongoDB...")
    subprocess.Popen([MONGO_DB_PATH, "--dbpath", DB_DATA_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    st.success("MongoDB started successfully!")

db = client["infact_db"]
hypotheses_collection = db["hypotheses"]
fs = GridFS(db)

# Function to preload evidence
def preload_evidence():
    identifier = "example_hypotheses_HumanGHGs_GlobalWarming"
    hypothesis_text = "Human-generated GHG emissions significantly increase global temperatures"
    
    evidence_files = [
        "/home/rrao/projects/InFactGAIA/hypotheses/example_hypotheses_HumanGHGs_GlobalWarming/evidence/greenhouse-gases.html",
        "/home/rrao/projects/InFactGAIA/hypotheses/example_hypotheses_HumanGHGs_GlobalWarming/evidence/World of Change_ Global Temperatures.html"
    ]

    # Check if the document exists and contains the hypothesis field
    document = hypotheses_collection.find_one({"identifier": identifier}, {"_id": 0, "evidence": 1, "hypothesis": 1})

    if not document:
        # If the document doesn't exist, create it with the hypothesis and an empty evidence list
        document = {"evidence": [], "hypothesis": hypothesis_text}
        hypotheses_collection.insert_one({"identifier": identifier, "hypothesis": hypothesis_text, "evidence": []})
    elif "hypothesis" not in document or not document["hypothesis"]:
        # If the hypothesis field is missing, update it
        hypotheses_collection.update_one({"identifier": identifier}, {"$set": {"hypothesis": hypothesis_text}})

    existing_filenames = {fs.get(f_id).filename for f_id in document.get("evidence", []) if fs.exists(f_id)}

    for file_path in evidence_files:
        file_name = os.path.basename(file_path)
        if file_name not in existing_filenames:
            with open(file_path, "rb") as f:
                file_id = fs.put(f, filename=file_name, content_type="application/octet-stream")
                hypotheses_collection.update_one({"identifier": identifier}, {"$push": {"evidence": file_id}}, upsert=True)


# Preload default evidence
preload_evidence()

# Load existing hypotheses
existing_hypotheses = {
    h["identifier"]: h.get("hypothesis", "No Hypothesis Available")  # Avoids KeyError
    for h in hypotheses_collection.find({}, {"_id": 0, "identifier": 1, "hypothesis": 1})
}


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

# Ensure identifier is unique
existing_entry = hypotheses_collection.find_one({"$or": [{"identifier": identifier}, {"hypothesis": hypothesis}]})
if existing_entry and existing_entry.get("identifier") != identifier:
    st.error("This hypothesis already exists with a different identifier!")
elif existing_entry and existing_entry.get("hypothesis") != hypothesis:
    st.error("This identifier is already linked to a different hypothesis!")
else:
    st.warning("Identifier already exists! You can add new evidence or process existing evidence.")

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

st.subheader("Upload Evidence")

document = hypotheses_collection.find_one({"identifier": identifier})
evidence_count = len(document["evidence"]) if document and document.get("evidence") else 0
st.write(f"Currently loaded evidence files: {evidence_count}")
uploaded_file = st.file_uploader("Upload document:", type=["pdf", "csv", "png", "html", "txt"])

def upload_evidence(identifier, file):
    if not identifier.strip():
        st.error("Please enter a Hypothesis Identifier before uploading evidence.")
        return

    # Ensure the identifier exists in MongoDB
    document = hypotheses_collection.find_one({"identifier": identifier}, {"evidence": 1})
    
    if not document:
        st.error("Hypothesis Identifier not found! Save it first.")
        return

    # Ensure local folder exists
    evidence_folder = f"./hypotheses/{identifier}/evidence/"
    os.makedirs(evidence_folder, exist_ok=True)

    # Check for duplicate file in GridFS
    existing_filenames = {fs.get(f_id).filename for f_id in document.get("evidence", []) if fs.exists(f_id)}

    if file.name in existing_filenames:
        st.warning(f"File '{file.name}' is already uploaded!")
        return

    # Save file locally in evidence folder
    local_file_path = os.path.join(evidence_folder, file.name)
    with open(local_file_path, "wb") as f:
        f.write(file.getbuffer())

    # Upload to GridFS
    file_data = io.BytesIO(file.getbuffer())
    file_id = fs.put(file_data, filename=file.name, content_type=file.type)

    # Update MongoDB record
    hypotheses_collection.update_one({"identifier": identifier}, {"$push": {"evidence": file_id}})
    
    st.success(f"Uploaded {file.name} to MongoDB GridFS and saved locally in {evidence_folder}!")

if uploaded_file:
    upload_evidence(identifier, uploaded_file)

log_placeholder = st.empty()  # UI placeholder for logs

class StreamToLogger(io.StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.log = ""

    def write(self, message):
        self.log += message
        self.placeholder.text_area("Logs:", self.log, height=400)  # Use text_area for persistent display

    def flush(self):
        pass  # Ensures logs are displayed properly

if st.button("Process Evidence"):
    if not identifier.strip() or not api_key.strip():
        st.error("⚠️ Please enter a Hypothesis Identifier and API Key!")
    else:
        hypothesis_folder_name = identifier
        results_dir = Path("./hypotheses") / hypothesis_folder_name / "results"
        os.makedirs(results_dir, exist_ok=True)

        # Redirect stdout and stderr to display logs in UI
        output_buffer = StreamToLogger(log_placeholder)
        sys.stdout, sys.stderr = output_buffer, output_buffer  

        try:
            print("🚀 Processing started...\n")
            process_evidence(node_type, hypothesis_folder_name, Path("./hypotheses"), api_key, model, hypothesis)
            print("✅ Processing completed!\n")

            latest_html_file = get_latest_html_path(node_type, results_dir)

            if latest_html_file and os.path.exists(latest_html_file):
                print(f"📄 Found latest analysis file: {latest_html_file}")

                with open(latest_html_file, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # Save to MongoDB GridFS
                file_id = fs.put(html_content.encode(), filename=os.path.basename(latest_html_file), content_type="text/html")

                # Update database with latest analysis file
                hypotheses_collection.update_one({"identifier": identifier}, {"$set": {"latest_analysis_result": file_id}})
                st.success("📊 Evidence processed successfully! Results stored.")

                # Force UI to refresh after successful processing
                st.rerun()
            else:
                st.warning("⚠️ No analysis result found!")

        except Exception as e:
            print(f"❌ Error: {e}")
            st.error(f"Error processing evidence: {e}")

        finally:
            # Restore normal stdout and stderr behavior
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__


st.subheader("Results")

document = hypotheses_collection.find_one({"identifier": identifier})

def retrieve_file_from_db(file_id):
    try:
        file_data = fs.get(file_id)
        return file_data.read(), file_data.filename, file_data.content_type
    except gridfs.errors.NoFile:
        st.error("File not found in MongoDB.")
    except Exception as e:
        st.error(f"Unexpected error retrieving file: {e}")
    return None, None, None

if document and document.get("latest_analysis_result"):
    file_id = document["latest_analysis_result"]

    try:
        file_content, filename, content_type = retrieve_file_from_db(file_id)

        if file_content:
            st.write(f"Latest Analysis Results - {filename}:")

            if content_type == "text/html":
                # ✅ Provide a download button for the HTML file
                st.download_button(label="Download Full Analysis", data=file_content, file_name=filename, mime="text/html")

                # ✅ Directly render the full HTML inside Streamlit (FULL PAGE, NO BOX!)
                st.components.v1.html(file_content.decode(), height=1200, scrolling=True)

            else:
                st.download_button(label="Download Analysis", data=file_content, file_name=filename)

    except Exception as e:
        st.error(f"Error displaying results: {e}")

else:
    st.info("No processed analysis results available yet.")
