import time
import streamlit as st
import sys
import os
# Add the src folder to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.db import hypothesis_collection

def check_hypothesis_id():
    """Checks if the user-entered ID exists in MongoDB and stores the result."""
    _id = st.session_state.get("hypothesis_id_input", "").strip()
    if not _id:
        st.session_state["id_exists"] = None
        return

    # Look up in DB
    existing = hypothesis_collection.find_one({"_id": _id})
    st.session_state["id_exists"] = True if existing else False


def enter_hypothesis():
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

                # Mark as created
                st.session_state["hypothesis_id"] = hypothesis_id

                # 🔥 FORCE CHECK so button disappears on rerun
                check_hypothesis_id()

                st.success(f"✅ Created new hypothesis with ID '{hypothesis_id}'")
                time.sleep(1)
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
                # Use a single set operation for the session state
                st.session_state["hypothesis_id"] = hypothesis_id
                
                # Also store the hypothesis text for later use
                hypothesis_doc = hypothesis_collection.find_one({"_id": hypothesis_id})
                if hypothesis_doc:
                    st.session_state["hypothesis_text"] = hypothesis_doc["text"]
                
                st.session_state.process_step = 3
                st.rerun()