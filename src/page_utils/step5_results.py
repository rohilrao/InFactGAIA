import streamlit as st

def display_results_step(db, fs, hypothesis_collection):
    """
    Handles Step 7: View Results
    
    Args:
        db: MongoDB database connection
        fs: GridFS instance
        hypothesis_collection: MongoDB collection for hypotheses
        
    Returns:
        str: Navigation action - "back", "home", or None
    """
    st.header("Step 7: View All Results")
    
    # Get hypothesis information
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
        st.stop()

    # Get the current hypothesis text from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
    st.write(f"**Hypothesis:** {hypothesis_text}")
    
    # Display summary
    if "auto_summary" in hypothesis_entry and hypothesis_entry["auto_summary"]:
        with st.expander("Hypothesis Summary", expanded=True):
            st.markdown(hypothesis_entry["auto_summary"])
    
    # Find all processed files for this hypothesis
    processed_files = list(db.fs.files.find({
        "metadata.hypothesis_id": hypothesis_id, 
        "status": "processed"
    }))
    
    if not processed_files:
        st.info("No processed evidence files found for this hypothesis yet.")
    else:
        st.subheader(f"Evidence Files ({len(processed_files)})")
        
        # Create tabs for each processed file
        if len(processed_files) > 0:
            tabs = st.tabs([file["filename"] for file in processed_files])
            
            for i, (tab, file) in enumerate(zip(tabs, processed_files)):
                with tab:
                    st.write(f"**File:** {file['filename']}")
                    st.write(f"**Processed on:** {file.get('last_processed_at', 'Unknown')}")
                    
                    # Check if this file has analysis results
                    if "analysis_file_id" in file:
                        analysis_id = file["analysis_file_id"]
                        if fs.exists(analysis_id):
                            # Display analysis
                            st.write("### Analysis Results")
                            file_content = fs.get(analysis_id).read().decode()
                            
                            # Download button
                            st.download_button(
                                label="⬇️ Download Analysis Report",
                                data=file_content,
                                file_name=f"{file['filename']}_analysis.html",
                                mime="text/html"
                            )
                            
                            # Display rendered HTML
                            st.components.v1.html(file_content, height=500, scrolling=True)
                    else:
                        st.warning("No analysis results available for this file.")
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Evidence Processing"):
            return "back"
    
    with col2:
        if st.button("Start New Session"):
            # Clear session state and start over
            for key in st.session_state.keys():
                if key != "process_step":  # Keep this to navigate
                    del st.session_state[key]
            
            return "home"
    
    return None  # No action taken