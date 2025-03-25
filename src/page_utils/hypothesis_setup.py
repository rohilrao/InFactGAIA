import streamlit as st
import time

def display_combined_hypothesis_step(hypothesis_collection, call_llm):
    """
    Combined function for hypothesis setup and summary in a single step.
    
    Args:
        hypothesis_collection: MongoDB collection for hypotheses
        call_llm: Function to call LLM API
        
    Returns:
        str: Navigation action - "back", "next", or None
    """
    st.header("Hypothesis Setup and Summary")

    # Initialize session state variables if they don't exist
    if "id_exists" not in st.session_state:
        st.session_state["id_exists"] = None
    if "summary_expanded" not in st.session_state:
        st.session_state["summary_expanded"] = False

    # Function to check if ID exists in database
    def check_hypothesis_id():
        _id = st.session_state.get("hypothesis_id_input", "").strip()
        if not _id:
            st.session_state["id_exists"] = None
            return

        # Look up in DB
        existing = hypothesis_collection.find_one({"_id": _id})
        st.session_state["id_exists"] = True if existing else False

    # 1. Hypothesis ID Input
    with st.container():
        st.markdown("### Enter a short identifier (ID) for the hypothesis")
        st.text_input(
            "Hypothesis ID (short name)",
            key="hypothesis_id_input",
            on_change=check_hypothesis_id
        )

    # 2. Hypothesis Text Input/Display
    hypothesis_id = st.session_state.get("hypothesis_id_input", "").strip()
    loaded_text = None
    hypothesis_doc = None
    
    if hypothesis_id and st.session_state["id_exists"] is True:
        # The ID exists in DB => fetch its text
        hypothesis_doc = hypothesis_collection.find_one({"_id": hypothesis_id})
        if hypothesis_doc:
            loaded_text = hypothesis_doc["text"]
            st.session_state["hypothesis_text_input"] = loaded_text
            st.success(f"Loaded existing hypothesis with ID '{hypothesis_id}'.")
        else:
            st.error("Inconsistent state: ID exists but not found in DB.")
            st.stop()
    
    # Display appropriate text area based on ID existence
    with st.container():
        if st.session_state["id_exists"] is True and loaded_text:
            st.markdown("### Existing Hypothesis")
            st.text_area(
                "Hypothesis Text",
                value=loaded_text,
                disabled=True,
                height=100
            )
        else:
            if hypothesis_id and st.session_state["id_exists"] is False:
                st.warning("No existing hypothesis found for this ID. You can create a new one below.")
            
            st.markdown("### Enter Your Hypothesis")
            st.text_area(
                "Hypothesis Text",
                key="hypothesis_text_input",
                placeholder="Enter your hypothesis details here",
                height=100
            )
            
            # Create Hypothesis Button
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
                    st.session_state["hypothesis_text"] = new_text

                    # Force check so button disappears on rerun
                    check_hypothesis_id()

                    st.success(f"✅ Created new hypothesis with ID '{hypothesis_id}'")
                    time.sleep(1)
                    st.rerun()

    # 3. Generate/Display Summary
    # Only show if we have a valid hypothesis ID
    can_show_summary = (st.session_state["id_exists"] is True and hypothesis_id) or \
                         st.session_state.get("hypothesis_id") == hypothesis_id
    
    if can_show_summary:
        st.markdown("---")
        st.markdown("### Hypothesis Summary")
        
        # Use the active hypothesis ID
        active_id = st.session_state.get("hypothesis_id", hypothesis_id)
        
        # Retrieve the complete document
        if not hypothesis_doc:
            hypothesis_doc = hypothesis_collection.find_one({"_id": active_id})
        
        if hypothesis_doc:
            # Process for Yes/No reformulation if needed
            need_reformulation = False
            original_text = hypothesis_doc.get("original_text", hypothesis_doc["text"])
            
            if "original_text" not in hypothesis_doc or hypothesis_doc["original_text"] == hypothesis_doc["text"]:
                need_reformulation = True
            
            if need_reformulation:
                with st.spinner("Reformulating hypothesis as a yes/no question..."):
                    prompt_reformulate = (
                        f"Given this hypothesis:\n\n'{original_text}'\n\n"
                        "Rewrite/Reformulate it as a clear, concise Yes-No question. " 
                        "The answer to the reformulated question should be either 'Yes' or 'No'. "
                        "Keep your response brief — ONLY return the reformulated question, nothing else."
                    )
                    
                    # Call LLM
                    yes_no_formulation = call_llm(
                        provider=st.session_state["provider"],
                        model=st.session_state["model"],
                        api_key=st.session_state["api_key"],
                        prompt_text=prompt_reformulate
                    ).strip()
                    
                    # Validate the response
                    if not yes_no_formulation.endswith('?'):
                        yes_no_formulation = yes_no_formulation.rstrip('.') + '?'
                    
                    # Update DB
                    hypothesis_collection.update_one(
                        {"_id": active_id},
                        {"$set": {
                            "original_text": original_text,
                            "text": yes_no_formulation
                        }}
                    )
                    
                    # Refresh the data
                    hypothesis_doc["original_text"] = original_text
                    hypothesis_doc["text"] = yes_no_formulation
                    st.session_state["hypothesis_text"] = yes_no_formulation
            
            # Display the refined hypothesis
            st.markdown("#### Refined Question:")
            st.markdown(f"**{hypothesis_doc['text']}**")
            
            # Generate or retrieve summary
            if "auto_summary" not in hypothesis_doc or hypothesis_doc["auto_summary"] is None:
                with st.spinner("Generating hypothesis summary..."):
                    prompt_summary = (
                        f"Given this yes/no hypothesis question:\n\n'{hypothesis_doc['text']}'\n\n"
                        "Create a concise, structured summary with three sections:\n\n"
                        "1. Background: Provide a brief paragraph summarizing the current state of knowledge on this topic.\n\n"
                        "2. Key Controversies: List 2-3 main points of debate in the field regarding this question.\n\n"
                        "3. Relevant Evidence:\n"
                        "   - Evidence Supporting the Hypothesis\n"
                        "   - Evidence Against the Hypothesis\n\n"
                        "Make your response clear and well-organized without excessive use of emojis. Include citations where relevant."
                    )
                    
                    llm_response = call_llm(
                        provider=st.session_state["provider"],
                        model=st.session_state["model"],
                        api_key=st.session_state["api_key"],
                        prompt_text=prompt_summary
                    )
                    
                    hypothesis_collection.update_one(
                        {"_id": active_id},
                        {"$set": {"auto_summary": llm_response}}
                    )
                    
                    # Refresh the data
                    hypothesis_doc["auto_summary"] = llm_response
            
            # Display the summary in an expandable section
            with st.expander("View Complete Summary", expanded=st.session_state["summary_expanded"]):
                st.markdown(hypothesis_doc["auto_summary"])
                if st.button("Keep Expanded" if not st.session_state["summary_expanded"] else "Collapse Summary"):
                    st.session_state["summary_expanded"] = not st.session_state["summary_expanded"]
                    st.rerun()
    
    # 4. Navigation Buttons
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("← Back"):
            return "back"

    with col2:
        # Only enable Next button if we have a valid hypothesis
        if can_show_summary:
            if st.button("Next →"):
                st.session_state["hypothesis_id"] = hypothesis_id
                return "next"
    
    return None  # No action taken