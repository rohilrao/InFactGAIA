import streamlit as st

def display_summary_step(hypothesis_collection, call_llm):
    """
    Handles Step 3: Hypothesis Refinement & Summary
    
    Args:
        hypothesis_collection: MongoDB collection for hypotheses
        call_llm: Function to call LLM API
        
    Returns:
        str: Navigation action - "back", "next", or None
    """
    st.header("Step 3: Hypothesis Refinement & Summary")

    hypothesis_id = st.session_state.get("hypothesis_id")
    if not hypothesis_id:
        st.warning("No Hypothesis ID found. Please go back to Step 2.")
        st.stop()

    # Retrieve hypothesis from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.warning("Hypothesis not found in DB. Please go back and create one.")
        st.stop()

    # Display the current hypothesis in minimal form
    st.write("#### Current Hypothesis:")
    st.write(f"**{hypothesis_entry['text']}**")

    # Keep original text on hand for rewriting (if not already set)
    original_text = hypothesis_entry.get("original_text", hypothesis_entry["text"])

    # ─────────────────────────────────────────────────────────────────
    # Automatically Reformulate as Yes/No with Progress Indicator
    # ─────────────────────────────────────────────────────────────────
    need_reformulation = False
    if "original_text" not in hypothesis_entry:
        need_reformulation = True
    elif hypothesis_entry["original_text"] == hypothesis_entry["text"]:
        need_reformulation = True

    if need_reformulation:
        with st.spinner("🔄 Reformulating hypothesis..."):
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
            
            # Validate the response (basic check that it ends with a question mark)
            if not yes_no_formulation.endswith('?'):
                yes_no_formulation = yes_no_formulation.rstrip('.') + '?'
            
            # Update DB
            hypothesis_collection.update_one(
                {"_id": hypothesis_id},
                {"$set": {
                    "original_text": original_text,  # store original if not set
                    "text": yes_no_formulation       # store newly refined text
                }}
            )
            
            # Refresh the data in session state for use in Step 4
            st.session_state["hypothesis_text"] = yes_no_formulation
            
            # Refresh the data in current view
            hypothesis_entry["original_text"] = original_text
            hypothesis_entry["text"] = yes_no_formulation
    
    st.write("#### Refined Hypothesis:")
    st.write(f"**{hypothesis_entry['text']}**")

    # ─────────────────────────────────────────────────────────────────
    # Automatically Generate Summary with Stepwise Progress Indication
    # ─────────────────────────────────────────────────────────────────
    if "auto_summary" not in hypothesis_entry or hypothesis_entry["auto_summary"] is None:
        with st.spinner("🔄 Identifying current state of the field..."):
            prompt_summary = (
                f"Given this yes/no hypothesis question:\n\n'{hypothesis_entry['text']}'\n\n"
                "Please provide:\n"
                "1. **Summarize existing knowledge** (short crisp points, with sources and links if possible).\n"
                "2. **Highlight key controversies** (if any).\n"
                "3. **Discuss relevant data** under two headings:\n"
                "   - ✅ Evidence in Favor\n"
                "   - ❌ Potentially Refuting Evidence\n\n"
                "Keep it succinct, well-structured, and visually clear.\n\n"
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
            
            with st.spinner("🔄 Identifying key controversies..."):
                llm_response = call_llm(
                    provider=st.session_state["provider"],
                    model=st.session_state["model"],
                    api_key=st.session_state["api_key"],
                    prompt_text=prompt_summary
                )
                
            with st.spinner("🔄 Identifying supporting and refuting evidence..."):
                hypothesis_collection.update_one(
                    {"_id": hypothesis_id},
                    {"$set": {"auto_summary": llm_response}}
                )
                
                # Refresh the data
                hypothesis_entry["auto_summary"] = llm_response
    
    st.write("#### Summary:")
    st.write(hypothesis_entry["auto_summary"])

    # ─────────────────────────────────────────────────────────────────
    # Navigation at the Bottom
    # ─────────────────────────────────────────────────────────────────
    st.write("---")  # Just a horizontal rule to separate content from nav
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("← Back to Step 2"):
            return "back"

    with col2:
        if st.button("Next → to File Upload"):
            return "next"
            
    return None  # No action taken