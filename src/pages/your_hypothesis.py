import streamlit as st
from pymongo import MongoClient
import gridfs
import datetime
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi
import openai
from anthropic import Anthropic

# ------------------------------------------------
# 🔐 MongoDB Connection
# ------------------------------------------------
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["infact_db_v3"]
fs = gridfs.GridFS(db)
hypothesis_collection = db["hypotheses"]

# ------------------------------------------------
# Helper: LLM call function
# ------------------------------------------------
def call_llm(provider, model, api_key, prompt_text):
    """
    Function to call GPT or Anthropic.
    """
    # For GPT
    if provider == "GPT":
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.1
        )
        return response.choices[0].message.content

    # For Anthropic
    elif provider == "Anthropic":
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        return message.content[0].text

    else:
        raise ValueError("Unsupported provider")

# ------------------------------------------------
# Wizard Setup
# ------------------------------------------------
if "process_step" not in st.session_state:
    st.session_state.process_step = 1

st.title("Hypothesis Wizard")

# ----------------------------
# STEP 1: AI Model Config
# ----------------------------
if st.session_state.process_step == 1:
    st.header("Step 1: AI Model Configuration")
    
    provider = st.selectbox("Select Provider:", ["GPT", "Anthropic"])  # Only GPT or Anthropic
    model_options = {
        "GPT": ["gpt-4o"],   # Example GPT models
        "Anthropic": ["claude-3-5-sonnet"]   # Example Anthropic model
    }
    model = st.selectbox("Select Model:", model_options[provider])
    api_key = st.text_input("Enter API Key:", type="password")

    if st.button("Next →"):
        if not provider or not model or not api_key.strip():
            st.warning("Please fill in all fields before proceeding.")
        else:
            st.session_state["provider"] = provider
            st.session_state["model"] = model
            st.session_state["api_key"] = api_key
            st.session_state.process_step = 2
            st.rerun()

# ----------------------------
# STEP 2: Hypothesis Setup
# ----------------------------
elif st.session_state.process_step == 2:
    st.header("Step 2: Hypothesis Setup")

    st.markdown("""
    ### Enter a **short identifier** for the hypothesis you want to test.
    - If the hypothesis already exists, it will be retrieved.
    - Otherwise, you can create it by entering text below.
    """)

    # Input for Hypothesis ID
    hypothesis_id_input = st.text_input("Enter Hypothesis ID:")
    if st.button("Load / Create Hypothesis"):
        if hypothesis_id_input:
            st.session_state["hypothesis_id"] = hypothesis_id_input
            st.rerun()

    # If we have a stored hypothesis ID in session
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if hypothesis_id:
        st.subheader(f"Current Hypothesis ID: `{hypothesis_id}`")
        # 🔎 Check if hypothesis ID exists
        hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})

        if hypothesis_entry:
            st.info(f"**Existing Hypothesis:** {hypothesis_entry['text']}")
        else:
            st.warning("This Hypothesis ID does not exist yet. Please enter a short description below to create it.")
            new_hypothesis_text = st.text_area("Enter Hypothesis Text (Cannot be edited after saving):")
            if st.button("💾 Save Hypothesis"):
                if new_hypothesis_text.strip():
                    hypothesis_collection.insert_one({
                        "_id": hypothesis_id,
                        "text": new_hypothesis_text,
                        "auto_summary": None    # We'll fill in Step 3
                    })
                    st.success("✅ Hypothesis saved successfully!")
                    st.rerun()
                else:
                    st.warning("⚠️ Hypothesis text cannot be empty!")

    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back"):
            st.session_state.process_step = 1
            st.rerun()
    with col2:
        if hypothesis_id:
            # Only allow Next if we have a valid hypothesis
            if st.button("Next →", key="to_step_3"):
                st.session_state.process_step = 3
                st.rerun()

# ----------------------------
# STEP 3: Generate Summary
# ----------------------------
elif st.session_state.process_step == 3:
    st.header("Step 3: Generate & Store LLM Summary")

    hypothesis_id = st.session_state.get("hypothesis_id")
    if not hypothesis_id:
        st.warning("No Hypothesis ID found. Please go back to Step 2.")
    else:
        # Retrieve hypothesis from DB
        hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
        if not hypothesis_entry:
            st.warning("Hypothesis not found in DB. Please go back and create one.")
        else:
            st.write("**Current Hypothesis Text:** ", hypothesis_entry["text"])
            st.write("""
            **At this step**, we call the chosen LLM to produce:
            1) The hypothesis as a yes-no question  
            2) A background paragraph summarizing knowledge/controversies  
            3) What data might be relevant  
            """)

            if st.button("Generate LLM Summary"):
                # Build the prompt to pass to LLM
                prompt = (
                    f"Given this hypothesis:\n\n'{hypothesis_entry['text']}'\n\n"
                    "Please rewrite it as a yes-no question. Then provide a background "
                    "paragraph summarizing the state of knowledge and controversies "
                    "around this hypothesis, and finally discuss what data might be "
                    "relevant for increasing or decreasing belief in it.\n"
                )

                # Call the LLM
                llm_response = call_llm(
                    provider=st.session_state["provider"],
                    model=st.session_state["model"],
                    api_key=st.session_state["api_key"],
                    prompt_text=prompt
                )

                # Store in DB as auto_summary
                hypothesis_collection.update_one(
                    {"_id": hypothesis_id},
                    {"$set": {"auto_summary": llm_response}}
                )

                st.success("✔️ Summary generated and saved to DB!")
                # Show the summary
                st.write("**LLM Summary:**")
                st.write(llm_response)

            # If there's already a summary, show it
            if hypothesis_entry.get("auto_summary"):
                st.write("**Existing LLM Summary:**")
                st.write(hypothesis_entry["auto_summary"])

    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Step 2"):
            st.session_state.process_step = 2
            st.rerun()
    with col2:
        if st.button("Next → to File Upload"):
            st.session_state.process_step = 4
            st.rerun()

# ----------------------------
# STEP 4: File Manager
# ----------------------------
elif st.session_state.process_step == 4:
    st.header("Step 4: Upload Files for Your Hypothesis")

    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
    else:
        st.subheader(f"Hypothesis ID: `{hypothesis_id}`")

        # ========== FILE UPLOAD SECTION (As in your original code) ========= #
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "png", "jpg", "html"])

        # ✅ If user removes file from the uploader, clear session state
        if "pending_upload" in st.session_state and not uploaded_file:
            del st.session_state["pending_upload"]

        if uploaded_file:
            # ✅ Store file details temporarily
            st.session_state["pending_upload"] = {
                "filename": uploaded_file.name,
                "content": uploaded_file.read()
            }
            st.success(f"✅ Ready to upload: {uploaded_file.name}")

        if "pending_upload" in st.session_state:
            file_name = st.session_state["pending_upload"]["filename"]
            # Check if this file already exists under the given hypothesis
            existing_file = fs.find_one({"hypothesis_id": hypothesis_id, "filename": file_name})

            if existing_file:
                st.warning(f"⚠️ A file named **{file_name}** already exists under this hypothesis.")
                del st.session_state["pending_upload"]  # Remove pending if duplicate found
            else:
                if st.button("Confirm Upload"):
                    file_id = fs.put(
                        st.session_state["pending_upload"]["content"],
                        filename=file_name,
                        hypothesis_id=hypothesis_id,
                        upload_date=str(datetime.date.today()),
                        status="unprocessed"
                    )
                    del st.session_state["pending_upload"]
                    st.session_state["last_uploaded_time"] = datetime.datetime.now(datetime.timezone.utc)
                    st.session_state["last_uploaded_time"] = datetime.datetime.utcnow()
                    st.success(f"Uploaded: {file_name}")
                    st.rerun()

        # 📂 Fetch & Display existing files
        files = list(fs.find({"hypothesis_id": hypothesis_id}))
        if files:
            st.subheader("Existing Files")
            for file in files:
                file_id = file._id
                filename = file.filename
                status = file.status
                status_display = (
                    '<span style="color: green; font-weight: bold;">Processed</span>'
                    if status == "processed" else
                    '<span style="color: red; font-weight: bold;">Unprocessed</span>'
                )

                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"📄 **{filename}** - {status_display}", unsafe_allow_html=True)
                with col2:
                    with fs.get(file_id) as grid_out:
                        file_content = grid_out.read()
                    st.download_button("⬇️ Download", file_content, filename, key=f"download_{file_id}")
                if status == "unprocessed":
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{file_id}"):
                            fs.delete(ObjectId(file_id))
                            st.warning(f"Deleted {filename}")
                            st.rerun()
        else:
            st.info("⚠️ No files found for this hypothesis.")

    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Step 3"):
            st.session_state.process_step = 3
            st.rerun()
    with col2:
        if st.button("Finish"):
            st.success("All steps completed!")
            # You could reset the wizard or navigate away:
            # st.session_state.process_step = 1
            # st.rerun()
