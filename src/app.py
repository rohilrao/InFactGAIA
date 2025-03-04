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
db = client["file_management"]
fs = gridfs.GridFS(db)

st.title("📂 File Explorer for MongoDB Atlas")

# 🌟 Enter User ID and Confirm
user_id_input = st.text_input("Enter User ID")
load_files = st.button("🔍 Load Files")

if load_files and user_id_input:
    st.session_state["user_id"] = user_id_input  # Store user ID in session state

# Get stored user ID
user_id = st.session_state.get("user_id", None)

if user_id:
    st.subheader(f"📂 Files for User ID: {user_id}")

    # 📤 File Upload Section
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg"])
    if uploaded_file:
        file_id = fs.put(
            uploaded_file.read(),
            filename=uploaded_file.name,
            user_id=user_id,
            upload_date=str(datetime.date.today()),
            status="unprocessed"
        )
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.experimental_rerun()  # Refresh to display updated files

    # 📂 Fetch and Display Uploaded Files
    files = list(fs.find({"user_id": user_id}))

    if files:
        for file in files:
            file_id = file._id
            filename = file.filename
            status = file.status

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 **{filename}** - `{status}`")
            with col2:
                st.download_button("⬇️ Download", file.read(), filename)
            with col3:
                if st.button("🗑️ Delete", key=str(file_id)):
                    fs.delete(ObjectId(file_id))
                    st.warning(f"Deleted {filename}")
                    st.experimental_rerun()
    else:
        st.write("⚠️ No files found for this user.")

