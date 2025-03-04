import streamlit as st
from pymongo import MongoClient
import gridfs
from pymongo.server_api import ServerApi

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
                st.write(f"- {file.filename}")  # Display unprocessed file names

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

        api_key = st.text_input("Enter API Key:", type="password")  # Masked input

    else:
        st.warning("❌ No hypothesis found with this ID.")
