import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import gridfs
import datetime
from bson.objectid import ObjectId

# 🔐 Secure MongoDB Connection
@st.cache_resource
def get_db_client():
    """Creates and caches a MongoDB client connection."""
    MONGO_URI = st.secrets["MONGO_URI"]  # Load from Streamlit secrets
    return MongoClient(MONGO_URI, server_api=ServerApi('1'))


# Initialize MongoDB
client = get_db_client()
db = client["file_management"]
fs = gridfs.GridFS(db)

st.title("📂 File Explorer for MongoDB Atlas")

# 📌 Select User ID
user_id = st.text_input("Enter User ID", "12345")

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

# 📂 Fetch and Display Uploaded Files
st.subheader("Uploaded Files")

# Get files for the given user ID
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
                st.experimental_rerun()  # Refresh the app after deletion

else:
    st.write("⚠️ No files found for this user.")
