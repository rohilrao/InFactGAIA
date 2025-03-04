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
fs = gridfs.GridFS(db)  # For file storage
hypothesis_collection = db["hypotheses"]

# 📌 Hypothesis Explorer Page
st.title("📜 Hypothesis Explorer")

# 📡 Fetch all stored hypotheses
hypotheses = list(hypothesis_collection.find({}))

if hypotheses:
    st.write(f"📊 **Total Hypotheses Stored:** {len(hypotheses)}")

    for hypothesis in hypotheses:
        hypothesis_id = hypothesis["_id"]
        hypothesis_text = hypothesis["text"]

        # 📂 Count the number of processed and unprocessed files for this hypothesis
        total_files = fs.find({"hypothesis_id": hypothesis_id})
        processed_files = sum(1 for file in total_files if file.status == "processed")
        total_files.rewind()  # Reset cursor
        unprocessed_files = sum(1 for file in total_files if file.status == "unprocessed")

        # 📌 Display Hypothesis ID
        st.markdown(f"### 🆔 Hypothesis ID: `{hypothesis_id}`")
        
        # 📌 Display Hypothesis Text
        st.write(f"**Hypothesis Text:**\n\n{hypothesis_text}")

        # 📊 File Status Summary
        st.write(f"📂 **Files Attached:** {processed_files + unprocessed_files} (✅ Processed: {processed_files} | ⏳ Unprocessed: {unprocessed_files})")

        st.markdown("---")  # Add a horizontal divider for clarity

else:
    st.warning("⚠️ No hypotheses found in the database.")
