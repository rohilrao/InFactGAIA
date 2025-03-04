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

class StreamToLogger(io.StringIO):
    """Redirects print statements to Streamlit UI."""
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.log = ""

    def write(self, message):
        self.log += message
        self.placeholder.text(self.log)  # Update the Streamlit UI log output

    def flush(self):
        pass  # No need to flush for Streamlit output



def load_or_create_node(node_type, hypothesis, model, api_key, hypothesis_id):
    """
    Load or create an InFact node state from MongoDB GridFS.
    - If an existing node state exists for the hypothesis, load it.
    - Otherwise, find the most recently processed file for the same hypothesis and use its node state.
    - If no node state exists, create a new node.
    """
    node_state = None

    # ✅ Check if there is already a node state for this hypothesis
    existing_node_state = db["fs.files"].find_one(
        {"hypothesis_id": hypothesis_id, "node_state_file_id": {"$exists": True}},
        sort=[("last_processed_at", -1)]  # Sort by most recent processing
    )

    if existing_node_state and "node_state_file_id" in existing_node_state:
        node_state_file_id = existing_node_state["node_state_file_id"]

        if fs.exists(ObjectId(node_state_file_id)):
            print(f"🔄 Loading existing node state from file ID: {node_state_file_id}")
            file_obj = fs.get(ObjectId(node_state_file_id))
            node_state = json.loads(file_obj.read().decode())  # Read JSON state

    # ✅ If no existing node state was found, create a new one
    if not node_state:
        print(f"✨ Creating new {node_type} InFact node for hypothesis ID: {hypothesis_id}...")
        node_state = {}  # Empty state for new node

    # ✅ Initialize the correct type of InFact node
    if node_type.lower() == "anthropic":
        node = AnthropicInFactNode(hypothesis=hypothesis, api_key=api_key, model=model)
    elif node_type.lower() == "gpt":
        node = GptInFactNode(hypothesis=hypothesis, api_key=api_key, model=model)
    elif node_type.lower() == "deepseek":
        node = DeepSeekInFactNode(hypothesis=hypothesis, api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown node type: {node_type}")

    return node

def process_evidence(node_type, hypothesis_id, api_key, model, hypothesis_text, log_output=None):
    """
    Processes all unprocessed evidence files for a given hypothesis.
    Works with MongoDB GridFS, ensuring files are handled via file paths.
    """
    hypotheses_collection = db["hypotheses"]
    files_collection = db["fs.files"]

    # Redirect logs if log_output is provided (e.g., in Streamlit)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    if log_output:
        sys.stdout, sys.stderr = log_output, log_output

    try:
        print("🚀 Processing started...\n")

        # ✅ Fetch unprocessed files for the hypothesis
        unprocessed_files = list(files_collection.find({"hypothesis_id": hypothesis_id, "status": "unprocessed"}))

        if not unprocessed_files:
            print(f"🚫 No unprocessed files found for hypothesis ID: {hypothesis_id}.")
            return

        print(f"📂 Found {len(unprocessed_files)} unprocessed files in GridFS.")

        # ✅ Load or create the node
        node = load_or_create_node(node_type, hypothesis_text, model, api_key, hypothesis_id)

        processed_files = []

        for file_meta in unprocessed_files:
            try:
                file_id = file_meta["_id"]
                filename = file_meta["filename"]

                print(f"🔄 Processing file: {filename}...")

                # ✅ Fetch file content from GridFS
                file_obj = fs.get(file_id)
                file_content = file_obj.read()
                file_extension = os.path.splitext(filename)[1].lower()

                # ✅ Create a temporary file for the evidence
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file_path = tmp_file.name
                    tmp_file.write(file_content)

                print(f"📂 Temporary evidence file saved at: {tmp_file_path}")

                # ✅ Process file using node (file path required)
                node.process_data(tmp_file_path)

                # ✅ Create a temporary JSON file for node state
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_node_file:
                    temp_node_state_path = tmp_node_file.name

                # ✅ Save node state to the temporary file using node's save() method
                node.save(temp_node_state_path)

                # ✅ Read the saved JSON content
                with open(temp_node_state_path, "r", encoding="utf-8") as f:
                    node_state_content = f.read()

                # ✅ Upload the node state to MongoDB GridFS
                state_file_id = fs.put(node_state_content.encode(), filename=f"{node_type}_state_{file_id}.json", content_type="application/json")

                # ✅ Cleanup node state temp file
                os.remove(temp_node_state_path)

                # ✅ Create a temporary file for analysis output
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_analysis_file:
                    temp_analysis_path = tmp_analysis_file.name

                # ✅ Render analysis output to the temporary file
                renderer = InFactRenderer()
                renderer.render_analysis(node, temp_analysis_path)

                # ✅ Read the analysis content
                with open(temp_analysis_path, "r", encoding="utf-8") as f:
                    analysis_content = f.read()

                # ✅ Store analysis in MongoDB GridFS
                processed_file_id = fs.put(
                    analysis_content.encode(),
                    filename=f"analysis_{Path(filename).stem}.html",
                    content_type="text/html"
                )

                # ✅ Cleanup analysis temp file
                os.remove(temp_analysis_path)

                # ✅ Ensure `processed_file_id` and `state_file_id` are valid before updating DB
                if processed_file_id and state_file_id:
                    files_collection.update_one(
                        {"_id": file_id},
                        {
                            "$set": {
                                "status": "processed",
                                "node_state_file_id": state_file_id,
                                "last_processed_at": datetime.datetime.utcnow(),
                                "analysis_file_id": processed_file_id,
                            }
                        }
                    )

                    print(f"✅ Analysis result stored in MongoDB with file_id: {processed_file_id}")

                # ✅ Cleanup temporary evidence file
                os.remove(tmp_file_path)

                processed_files.append(filename)

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")
                # ✅ File status remains 'unprocessed' if there's an error

        if processed_files:
            print(f"✅ Successfully processed {len(processed_files)} evidence files.")
        else:
            print(f"🚫 No new evidence was processed.")

        print("✅ Processing completed!")

    except Exception as e:
        print(f"❌ Error during processing: {e}")

    finally:
        # Restore normal stdout and stderr behavior
        sys.stdout, sys.stderr = original_stdout, original_stderr
        
# 🔐 MongoDB Connection
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["hypothesis_management"]
fs = gridfs.GridFS(db)  # File storage system
hypothesis_collection = db["hypotheses"]

# 📌 Hypothesis Processor Page
st.title("⚙️ Hypothesis Processor")

# 🌟 Enter Hypothesis ID
hypothesis_id = st.text_input("Enter Hypothesis ID:")
load_hypothesis = st.button("🔍 Load Hypothesis")

if load_hypothesis and hypothesis_id:
    st.session_state["hypothesis_id"] = hypothesis_id  # Store hypothesis ID

# Get stored hypothesis ID
hypothesis_id = st.session_state.get("hypothesis_id", None)

if hypothesis_id:
    # 📡 Fetch Hypothesis
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})

    if hypothesis_entry:
        st.subheader(f"📑 Hypothesis for ID: `{hypothesis_id}`")
        st.write(f"**Hypothesis Text:**\n\n{hypothesis_entry['text']}")

        # 📂 Get Unprocessed Files
        unprocessed_files = list(fs.find({"hypothesis_id": hypothesis_id, "status": "unprocessed"}))

        if unprocessed_files:
            st.write(f"📂 **Unprocessed Files:** {len(unprocessed_files)}")
            for file in unprocessed_files:
                file_id = file._id
                filename = file.filename
                st.write(f"- **File:** {filename}")
        else:
            st.warning("⚠️ No unprocessed files found. Please navigate to the **Hypothesis Manager** to upload new files.")

        # ⚙️ Node Type, Model Selection, and API Key Input
        st.subheader("⚙️ Processing Configuration")

        node_type = st.selectbox("Select Node Type:", ["GPT", "Anthropic", "DeepSeek"])
        model_options = {
            "GPT": ["chatgpt-4o-latest"],
            "Anthropic": ["claude-3-5-sonnet-20241022"],
            "DeepSeek": ["deepseek-chat"]
        }
        model = st.selectbox("Select AI Model:", model_options[node_type])
        api_key = st.text_input("Enter API Key:", type="password")

        # 🚀 Process Evidence Files
        if unprocessed_files:
            if st.button("🚀 Process Evidence Files"):
                log_placeholder = st.empty()  # Create placeholder for logs
                output_buffer =StreamToLogger(log_placeholder)
                sys.stdout, sys.stderr = output_buffer, output_buffer  # Redirect logs to UI

                try:
                    process_evidence(
                        node_type=node_type,
                        hypothesis_id=hypothesis_id,
                        api_key=api_key,
                        model=model,
                        hypothesis_text=hypothesis_entry["text"],
                        log_output=output_buffer
                    )
                except Exception as e:
                    st.error(f"❌ Error processing evidence: {e}")
                finally:
                    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        else:
            st.warning("⚠️ No unprocessed files available for processing.")

        # ✅ Display Processed Analysis Results
        st.subheader("📊 Processed Files & Results")

        processed_files = list(fs.find({"hypothesis_id": hypothesis_id, "status": "processed"}))

        if processed_files:
            for file in processed_files:
                file_id = file._id
                filename = file.filename
                node_state_id = file.get("node_state_file_id", None)
                last_processed = file.get("last_processed_at", None)
                analysis_file_id = file.get("analysis_file_id", None)

                st.write(f"### **📂 File: {filename}**")

                if last_processed:
                    st.write(f"🕒 **Last Processed:** {last_processed}")

                if node_state_id:
                    if fs.exists(node_state_id):
                        st.download_button(
                            label="⬇️ Download Node State",
                            data=fs.get(node_state_id).read(),
                            file_name=f"{filename}_node_state.json",
                            mime="application/json"
                        )

                if analysis_file_id:
                    if fs.exists(analysis_file_id):
                        file_content = fs.get(analysis_file_id).read().decode()
                        st.download_button(
                            label="⬇️ Download Analysis Report",
                            data=file_content,
                            file_name=f"{filename}_analysis.html",
                            mime="text/html"
                        )
                        st.components.v1.html(file_content, height=600, scrolling=True)

                st.markdown("---")
        else:
            st.info("No processed files available yet.")

    else:
        st.warning("❌ No hypothesis found with this ID.")
