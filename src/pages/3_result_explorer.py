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
db = client["infact_db_v3"]
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
st.title("Result Explorer")

# Input for Hypothesis ID
hypothesis_id = st.text_input("Enter Hypothesis ID:")

# Button to load results
if st.button("Load Results"):
    st.session_state["loaded_hypothesis_id"] = hypothesis_id  # Store ID in session

# Get stored Hypothesis ID after button click
loaded_hypothesis_id = st.session_state.get("loaded_hypothesis_id", None)

if loaded_hypothesis_id:
    data = query_hypothesis(loaded_hypothesis_id)

    if data:
        #st.subheader("📑 Hypothesis Details")
        st.write(f"**Hypothesis ID:** `{loaded_hypothesis_id}`")
        st.write(f"**Hypothesis Text:** {data.get('text', 'No text available')}")

        # 🔍 Fetch associated files from GridFS
        files = list(fs.find({"hypothesis_id": loaded_hypothesis_id}))
        file_count = len(files)

        st.write(f"📂 **Number of Associated Evidence Files:** {file_count}")

        # 🔍 Fetch results (processed files) & download analysis
        st.subheader("Results")
        results = list(fs.find({"hypothesis_id": loaded_hypothesis_id, "status": "processed"}))
        if results:
            for file in results:
                file_id = file._id
                filename = file.filename
                analysis_file_id = getattr(file, "analysis_file_id", None)

                st.write(f"**{filename}**")

                # ✅ Check if the file has an associated analysis file
                if analysis_file_id and fs.exists(ObjectId(analysis_file_id)):
                    analysis_content = fs.get(ObjectId(analysis_file_id)).read()

                    st.download_button(
                        label="⬇️ Download Analysis Report",
                        data=analysis_content,
                        file_name=f"analysis_{filename}.html",
                        mime="text/html"
                    )
                else:
                    st.warning(f"⚠️ No analysis report found for {filename}.")
        else:
            st.info("⚠️ No processed results available.")
