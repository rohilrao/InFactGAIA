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
db = client["infact_db_v3"]
fs = gridfs.GridFS(db)
hypothesis_collection = db["hypotheses"]

# ✅ Reset hypothesis state when navigating away
current_page = "hypothesis_manager"
if "previous_page" in st.session_state and st.session_state["previous_page"] != current_page:
    st.session_state.pop("hypothesis_id", None)  # Clear stored hypothesis ID
    st.session_state.pop("pending_upload", None)  # Clear pending uploads
st.session_state["previous_page"] = current_page  # Update session state

st.title("📂 Hypothesis File Manager")

# 🌟 Hypothesis ID Input
st.markdown("### Enter a **short identifier** for the hypothesis you want to test.")
st.markdown("*If the hypothesis already exists, this page will retrieve it. Otherwise, you can create it here.*")
st.markdown("🔎 *All saved hypotheses can be explored in the **Hypothesis Explorer** page.*")

hypothesis_id_input = st.text_input("Enter Hypothesis ID:")
load_hypothesis = st.button("Load Hypothesis")

if load_hypothesis and hypothesis_id_input:
    st.session_state["hypothesis_id"] = hypothesis_id_input  # Store hypothesis ID in session

# Get stored hypothesis ID
hypothesis_id = st.session_state.get("hypothesis_id", None)

if hypothesis_id:
    st.subheader(f"Current Hypothesis ID: `{hypothesis_id}`")

    # 🔎 Check if hypothesis ID exists
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})

    if hypothesis_entry:
        st.info(f"**Hypothesis:**\n\n{hypothesis_entry['text']}")
    else:
        # 🌱 Allow user to enter a new hypothesis text
        st.warning("⚠️ This Hypothesis ID does not exist. Please enter a short description and save it.")
        new_hypothesis_text = st.text_area("Enter Hypothesis Text (Cannot be edited after saving):")

        if st.button("💾 Save Hypothesis"):
            if new_hypothesis_text.strip():
                hypothesis_collection.insert_one({"_id": hypothesis_id, "text": new_hypothesis_text})
                st.success("✅ Hypothesis saved successfully! (Now it cannot be edited)")
                st.rerun()
            else:
                st.warning("⚠️ Hypothesis text cannot be empty!")

    # 📤 File Upload Section
    st.subheader(f"📂 Upload Files for Hypothesis ID: `{hypothesis_id}`")

    # ✅ Step 1: Store uploaded file in session state before committing to MongoDB
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg", "html"])

    # ✅ If the user removes the uploaded file (presses ❌), clear session state
    if "pending_upload" in st.session_state and not uploaded_file:
        del st.session_state["pending_upload"]

    if uploaded_file:
        # ✅ Store file details temporarily before committing
        st.session_state["pending_upload"] = {
            "filename": uploaded_file.name,
            "content": uploaded_file.read()
        }
        st.success(f"✅ Ready to upload: {uploaded_file.name}")

    # ✅ Step 2: Confirm Upload (Avoids false duplicate detection)
    if "pending_upload" in st.session_state:
        file_name = st.session_state["pending_upload"]["filename"]

        # ✅ Check if this file already exists under the given hypothesis
        existing_file = fs.find_one({"hypothesis_id": hypothesis_id, "filename": file_name})

        if existing_file:
            st.warning(f"⚠️ A file named **{file_name}** already exists under this hypothesis.")
            del st.session_state["pending_upload"]  # ✅ Remove pending upload if duplicate found
        else:
            if st.button("Confirm Upload"):
                file_id = fs.put(
                    st.session_state["pending_upload"]["content"],  # ✅ Read file content from session state
                    filename=file_name,
                    hypothesis_id=hypothesis_id,  # ✅ Ensures association with this hypothesis
                    upload_date=str(datetime.date.today()),
                    status="unprocessed"
                )

                # ✅ Clear session state **before** rerun to avoid duplicate check error
                del st.session_state["pending_upload"]
                st.session_state["last_uploaded_file"] = file_name  # Store last uploaded file
                st.session_state["last_uploaded_time"] = datetime.datetime.utcnow()  # Timestamp
                
                # ✅ Perform an immediate rerun
                st.rerun()


    # 📂 Fetch and Display Uploaded Files (Only if Files Exist)
    files = list(fs.find({"hypothesis_id": hypothesis_id}))

    if files:
        st.subheader("Existing Files")
        for file in files:
            file_id = file._id
            filename = file.filename
            status = file.status

            # ✅ Color-code status
            status_display = (
                '<span style="color: green; font-weight: bold;">Processed</span>'
                if status == "processed" else
                '<span style="color: red; font-weight: bold;">Unprocessed</span>'
            )

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"📄 **{filename}** - {status_display}", unsafe_allow_html=True)  # ✅ Apply color styling
            with col2:
                with fs.get(file_id) as grid_out:
                    file_content = grid_out.read()
                st.download_button("⬇️ Download", file_content, filename, key=f"download_{file_id}")
            if status == "unprocessed":
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{file_id}"):  # Unique key for delete button
                        fs.delete(ObjectId(file_id))
                        st.warning(f"Deleted {filename}")
                        st.rerun()

    else:
        st.info("⚠️ No files found for this hypothesis.")
