

import streamlit as st
from pymongo import MongoClient
import gridfs
import datetime
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

st.title("📂 Hypothesis File Manager")

# 🌟 Enter hypothesis ID and Confirm
hypothesis_id_input = st.text_input("Enter Hypothesis ID:")
load_hypothesis = st.button("🔍 Load Hypothesis")

if load_hypothesis and hypothesis_id_input:
    st.session_state["hypothesis_id"] = hypothesis_id_input  # Store hypothesis ID in session state

# Get stored hypothesis ID
hypothesis_id = st.session_state.get("hypothesis_id", None)

if hypothesis_id:
    st.subheader(f"📑 Hypothesis for ID: {hypothesis_id}")

    # Check if hypothesis ID exists
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})

    if hypothesis_entry:
        st.info(f"**Hypothesis Text:**\n\n{hypothesis_entry['text']}")
    else:
        # Allow user to enter a hypothesis text if ID is new
        new_hypothesis_text = st.text_area("Enter Hypothesis Text (Cannot be edited after saving):")
        if st.button("💾 Save Hypothesis"):
            if new_hypothesis_text.strip():
                hypothesis_collection.insert_one({"_id": hypothesis_id, "text": new_hypothesis_text})
                st.success("✅ Hypothesis saved successfully! (Now it cannot be edited)")
                st.rerun()
            else:
                st.warning("⚠️ Hypothesis text cannot be empty!")

    # 📤 File Upload Section
    st.subheader(f"📂 Files for Hypothesis ID: {hypothesis_id}")
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg"])

    if uploaded_file:
        # Check for duplicates
        existing_file = fs.find_one({"hypothesis_id": hypothesis_id, "filename": uploaded_file.name})

        if existing_file:
            st.warning(f"⚠️ A file named **{uploaded_file.name}** already exists. Duplicate uploads are not allowed.")
        else:
            # Upload the new file if it's not a duplicate
            file_id = fs.put(
                uploaded_file.read(),
                filename=uploaded_file.name,
                hypothesis_id=hypothesis_id,
                upload_date=str(datetime.date.today()),
                status="unprocessed"
            )
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.rerun()  # Refresh to display updated files

    # 📂 Fetch and Display Uploaded Files
    files = list(fs.find({"hypothesis_id": hypothesis_id}))

    if files:
        for file in files:
            file_id = file._id
            filename = file.filename
            status = file.status

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 **{filename}** - `{status}`")
            with col2:
                st.download_button("⬇️ Download", file.read(), filename, key=f"download_{file_id}")  # Unique key added
            if status == "unprocessed":
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{file_id}"):  # Unique key for delete button
                        fs.delete(ObjectId(file_id))
                        st.warning(f"Deleted {filename}")
                        st.rerun()
    else:
        st.write("⚠️ No files found for this hypothesis.")
