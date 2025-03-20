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

def check_hypothesis_id():
    """Checks if the user-entered ID exists in MongoDB and stores the result."""
    _id = st.session_state.get("hypothesis_id_input", "").strip()
    if not _id:
        st.session_state["id_exists"] = None
        return

    # Look up in DB
    existing = hypothesis_collection.find_one({"_id": _id})
    st.session_state["id_exists"] = True if existing else False

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

#st.title("Hypothesis Wizard")

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
    ### Enter a short identifier (ID) for the hypothesis.
    - If the ID already exists, we will load it in read-only form.
    - If the ID does not exist, you can create a new hypothesis below.
    """)

    # Ensure we track existence checks
    if "id_exists" not in st.session_state:
        st.session_state["id_exists"] = None

    # 1) ID input with on_change to dynamically check the DB
    st.text_input(
        "Hypothesis ID (short name)",
        key="hypothesis_id_input",
        on_change=check_hypothesis_id  # your existing function; sets st.session_state["id_exists"] to True/False
    )

    # 2) Check existence
    hypothesis_id = st.session_state.get("hypothesis_id_input", "").strip()
    loaded_text = None

    if hypothesis_id and st.session_state["id_exists"] is True:
        # The ID exists in DB => fetch its text
        doc = hypothesis_collection.find_one({"_id": hypothesis_id})
        if doc:
            loaded_text = doc["text"]
        else:
            # If 'id_exists' = True but doc not found => inconsistent, but let's handle gracefully
            st.error("Inconsistent state: ID said to exist, but not found in DB.")
            st.stop()

        # Put the existing text in session_state for display (read-only)
        st.session_state["hypothesis_text_input"] = loaded_text

        st.info(f"Loaded existing hypothesis with ID '{hypothesis_id}'. You cannot overwrite it.")
    elif hypothesis_id and st.session_state["id_exists"] is False:
        st.warning("No existing hypothesis found for this ID. You can create a new one below.")
    else:
        st.write("Please enter an ID above to check availability.")

    # 3) Display text area
    #    If ID exists => show read-only text area with loaded_text
    #    If ID does not exist => editable text area for new text
    if st.session_state["id_exists"] is True and loaded_text:
        # Show read-only text area for existing ID
        st.markdown("#### Existing Hypothesis Text (read-only):")
        st.text_area(
            "Existing Hypothesis Text",
            value=loaded_text,
            disabled=True
        )
    else:
        # ID does not exist, so let user type a new text
        st.markdown("#### New Hypothesis Text (cannot be edited after creation):")
        st.text_area(
            "Enter your hypothesis details here",
            key="hypothesis_text_input"  # updatable in session
        )

    # 4) Navigation Buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # "← Back"
        if st.button("← Back"):
            st.session_state.process_step = 1
            st.rerun()

    with col2:
        # Only show "Create Hypothesis" if ID doesn't exist
        if st.session_state["id_exists"] is False and hypothesis_id:
            if st.button("Create Hypothesis"):
                # Must have text
                new_text = st.session_state.get("hypothesis_text_input", "").strip()
                if not new_text:
                    st.warning("⚠️ Please enter text before creating a new hypothesis.")
                    st.stop()

                # Insert new doc
                hypothesis_collection.insert_one({
                    "_id": hypothesis_id,
                    "original_text": new_text,
                    "text": new_text,
                    "auto_summary": None
                })
                st.session_state["hypothesis_id"] = hypothesis_id
                st.success(f"✅ Created new hypothesis with ID '{hypothesis_id}'")
                st.rerun()
        else:
            # If ID exists, user can't create or overwrite
            st.caption("No creation needed if ID already exists.")

    with col3:
        # "Next →" button
        # We only allow going forward if we have an existing or newly created hypothesis_id
        # i.e. either st.session_state["id_exists"] is True OR we've just created a new one
        already_in_db = (st.session_state["id_exists"] is True and hypothesis_id)
        newly_created = st.session_state.get("hypothesis_id") == hypothesis_id and hypothesis_id
        can_proceed = already_in_db or newly_created

        if can_proceed:
            if st.button("Next →"):
                # Save the "active" hypothesis ID in session, if not already set
                if "hypothesis_id" not in st.session_state:
                    st.session_state["hypothesis_id"] = hypothesis_id
                st.session_state.process_step = 3
                st.rerun()

# ----------------------------
# STEP 3: Generate Summary
# ----------------------------


elif st.session_state.process_step == 3:
    st.header("Step 3: Reformulate & Generate Summary")

    hypothesis_id = st.session_state.get("hypothesis_id")
    if not hypothesis_id:
        st.warning("No Hypothesis ID found. Please go back to Step 2.")
    else:
        # Retrieve hypothesis from DB
        hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
        if not hypothesis_entry:
            st.warning("Hypothesis not found in DB. Please go back and create one.")
        else:
            st.subheader("Part 1: Reformulate as Yes/No Question")
            st.write("**Current Hypothesis Text:** ", hypothesis_entry["text"])
            
            # Store the original text if not already stored
            original_text = hypothesis_entry.get("original_text", hypothesis_entry["text"])
            
            # FIRST LLM CALL - Reformulate to yes/no
            if st.button("Reformulate as Yes/No Question"):
                prompt_reformulate = (
                    f"Given this hypothesis:\n\n'{original_text}'\n\n"
                    "Rewrite it as a clear, concise Yes-No question. "
                    "Keep your response brief - ONLY return the reformulated question, nothing else."
                )
                
                # Call LLM for reformulation
                yes_no_formulation = call_llm(
                    provider=st.session_state["provider"],
                    model=st.session_state["model"],
                    api_key=st.session_state["api_key"],
                    prompt_text=prompt_reformulate
                ).strip()
                
                # Update DB - save original and update text
                hypothesis_collection.update_one(
                    {"_id": hypothesis_id},
                    {"$set": {
                        "original_text": original_text,
                        "text": yes_no_formulation
                    }}
                )
                
                st.success("✅ Hypothesis reformulated as Yes/No question!")
                st.write("**Reformulated Hypothesis:**", yes_no_formulation)
                st.rerun()  # Refresh to show updated state
            
            # PART 2 - Generate Summary (only shown after reformulation)
            if hypothesis_entry.get("original_text"):  # Only show if reformulation has been done
                st.subheader("Part 2: Generate Summary")
                st.write("**Yes/No Hypothesis:** ", hypothesis_entry["text"])
                
                # If there's already a summary, show it
                if hypothesis_entry.get("auto_summary"):
                    st.write("**Existing LLM Summary:**")
                    st.write(hypothesis_entry["auto_summary"])
                
                # SECOND LLM CALL - Generate detailed summary
                if st.button("Generate Detailed Summary"):
                    # Build the prompt for summary
                    prompt_summary = (
                        f"Given this yes/no hypothesis question:\n\n'{hypothesis_entry['text']}'\n\n"
                        "Please provide:\n"
                        "1. **Summarize existing knowledge** (short crisp points, with sources and clickable links if possible).\n"
                        "2. **Highlight key controversies** (if any).\n"
                        "3. **Discuss relevant data** under two subheadings:\n"
                        "   - ✅ Evidence in Favor\n"
                        "   - ❌ Potentially Refuting Evidence\n\n"
                        "Ensure the response is minimal, well-structured, and visually clear.\n\n"
                        "**Expected Output Format:**\n\n"
                        "### Existing Knowledge:\n"
                        "- 🔹 [Key fact 1] (Source)\n"
                        "- 🔹 [Key fact 2] (Source)\n"
                        "- 🔹 [Key fact 3] (Source)\n\n"
                        "### Controversies:\n"
                        "- ❗ [Main controversy 1]\n"
                        "- ❗ [Main controversy 2]\n\n"
                        "### Relevant Data:\n\n"
                        "#### ✅ Evidence in Favor:\n"
                        "- [Supporting evidence 1] (Study/Source)\n"
                        "- [Supporting evidence 2] (Study/Source)\n\n"
                        "#### ❌ Potentially Refuting Evidence:\n"
                        "- [Counter evidence 1] (Study/Source)\n"
                        "- [Counter evidence 2] (Study/Source)\n"
                    )
                    
                    # Call the LLM for summary
                    llm_response = call_llm(
                        provider=st.session_state["provider"],
                        model=st.session_state["model"],
                        api_key=st.session_state["api_key"],
                        prompt_text=prompt_summary
                    )
                    
                    # Store in DB, replace any existing summary
                    hypothesis_collection.update_one(
                        {"_id": hypothesis_id},
                        {"$set": {"auto_summary": llm_response}}
                    )
                    
                    st.success("✔️ Summary generated and saved!")
                    st.write("**LLM Summary:**")
                    st.write(llm_response)


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
