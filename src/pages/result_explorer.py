import streamlit as st
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi

# 🔐 MongoDB Connection
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["hypothesis_management"]
fs = gridfs.GridFS(db)
hypothesis_collection = db["hypotheses"]

# Function to query the MongoDB for hypothesis ID
def query_hypothesis(hypothesis_id):
    try:
        hypothesis = hypothesis_collection.find_one({"_id": hypothesis_id})
        if hypothesis:
            return hypothesis
        else:
            st.error("❌ Hypothesis not found")
            return None
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
        return None

# Streamlit app layout
st.title("📊 Result Explorer")

# Input for Hypothesis ID
hypothesis_id = st.text_input("Enter Hypothesis ID:")

if hypothesis_id:
    data = query_hypothesis(hypothesis_id)
    
    if data:
        st.subheader("📑 Hypothesis Details")
        st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
        st.write(f"**Hypothesis Text:** {data.get('text', 'No text available')}")

        # 🔍 Fetch associated files from GridFS
        st.subheader("📂 Associated Files")
        files = list(fs.find({"hypothesis_id": hypothesis_id}))
        if files:
            for file in files:
                st.write(f"📄 **{file.filename}** - `{file.status}`")
        else:
            st.info("⚠️ No associated files found.")

        # 🔍 Fetch results (processed files)
        st.subheader("📊 Results")
        results = list(fs.find({"hypothesis_id": hypothesis_id, "status": "processed"}))
        if results:
            for file in results:
                file_id = file._id
                filename = file.filename

                st.write(f"📊 **{filename}**")
                st.download_button(
                    label="⬇️ Download Result",
                    data=fs.get(file_id).read(),
                    file_name=filename,
                    mime="text/html"
                )
        else:
            st.info("⚠️ No processed results available.")
