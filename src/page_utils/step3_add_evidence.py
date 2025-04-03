import streamlit as st
import datetime
import time
from file_processor import process_file

def delete_file(db, fs, file_id):
    """Delete a file from GridFS and its metadata from the database"""
    try:
        # Delete from GridFS
        fs.delete(file_id)
        
        # Delete any metadata references
        db.fs.files.delete_one({"_id": file_id})
        db.fs.chunks.delete_many({"files_id": file_id})
        
        st.success("File deleted successfully")
        return True
    except Exception as e:
        st.error(f"Error deleting file: {str(e)}")
        return False

def display_file_upload_step(db, fs, hypothesis_collection):
    """
    Step 3: Add evidence to hypothesis by uploading files
    
    Args:
        db: MongoDB database connection
        fs: GridFS connection
        hypothesis_collection: MongoDB collection for hypotheses
        
    Returns:
        "next" to proceed to next step, "back" to return to previous step
    """
    # 1. HEADER SECTION
    st.markdown("### :orange[Upload Files for Your Hypothesis]")
    parsed_data = None
    
    # Get hypothesis information 
    hypothesis_id = st.session_state.get("hypothesis_id", None)

    # Check if we have hypothesis_id in session state
    if not hypothesis_id:
        # Also check hypothesis_id_input as fallback
        hypothesis_id = st.session_state.get("hypothesis_id_input", "").strip()
        if hypothesis_id:
            st.warning(f"Using hypothesis ID from input field: {hypothesis_id}")
            # Save it properly to session state
            st.session_state["hypothesis_id"] = hypothesis_id
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
                if st.button("← Back to Hypothesis Setup"):
                    return "back"
            st.stop()

    # Get the current hypothesis text from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.error(f"Could not find hypothesis with ID '{hypothesis_id}' in database.")
            if st.button("← Back to Hypothesis Setup"):
                return "back"
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.session_state["hypothesis_text"] = hypothesis_text

    # Simple hypothesis display
    st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
    st.write(f"**Hypothesis:** {hypothesis_text}")
    st.divider()
    
    # 2. EXISTING FILES SECTION
    st.markdown("### :orange[Current Files]")

    # Get existing files for this hypothesis, excluding node state and rendered hypothesis files
    existing_files = list(db.fs.files.find({
        "metadata.hypothesis_id": hypothesis_id,
        "filename": {"$not": {"$regex": "node_state|rendered_hypothesis"}}
    }))

    # Check for unprocessed or ready_for_analysis files
    unprocessed_files = [f for f in existing_files if f.get("status") == "unprocessed"]
    ready_for_analysis_files = [f for f in existing_files if f.get("status") == "ready_for_analysis" and 
                               f.get("parsing_complete", False)]
    processed_files = [f for f in existing_files if f.get("status") != "unprocessed" 
                      and f.get("status") != "processing"
                      and f.get("parsing_complete", False)]

    if existing_files:
        st.caption(f"{len(existing_files)} file(s) associated with this hypothesis")

        # Create a scrollable container
        file_container = st.container(height=250, border=True)

        # Display files within the scrollable container
        with file_container:
            for idx, file in enumerate(existing_files):
                file_id = file["_id"]
                filename = file["filename"]

                # Create a row for each file - add a column for download
                col1, col2, col3, col4 = st.columns([3, 3, 1, 1])

                # File name column
                with col1:
                    st.write(f"**{filename}**")
                    # Add a small display of file ID for debugging
                    st.caption(f"ID: {str(file_id)[-6:]}")

                # Status column
                with col2:
                    if file.get("status") == "ready_for_analysis" and file.get("parsing_complete", False):
                        st.markdown("<span style='color:green'>🔍 Ready for Analysis</span>", unsafe_allow_html=True)
                    elif file.get("status") == "processing":
                        st.markdown("<span style='color:orange'>⏳ Processing</span>", unsafe_allow_html=True)
                    elif file.get("status") == "processed":
                        st.markdown("<span style='color:orange'>✅ Processed</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#888'>⚪ Unprocessed</span>", unsafe_allow_html=True)

                # Download column
                with col4:
                    try:
                        # Retrieve file content from GridFS
                        file_content = fs.get(file_id).read()
                        
                        # Direct download button
                        st.download_button(
                            label="Download",
                            data=file_content,
                            file_name=filename,
                            mime="application/octet-stream",
                            key=f"download_{idx}",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error retrieving file: {str(e)}")

                # Action column (Delete)
                with col3:
                    if file.get("status") != "processed":
                        if st.button("Delete", key=f"delete_{idx}", use_container_width=True):
                            if delete_file(db, fs, file_id):
                                st.rerun()

                # Add a separator between files
                if idx < len(existing_files) - 1:
                    st.divider()
    else:
        st.info("No files uploaded yet. Upload your first file below.")

    st.divider()
    
    # 3. UPLOAD NEW FILE SECTION
    st.markdown("### :orange[Upload New File]")
    st.write("You can upload one file at a time. Please wait for processing to complete before uploading another file.")
    
    # File uploader
    uploaded_file = st.file_uploader("Select a file to upload", type=["txt", "pdf", "png", "jpg", "html", "csv"])

    # Logic for handling file upload
    if uploaded_file:
        file_name = uploaded_file.name
        print(f"DEBUG - User selected file: '{file_name}'")

        # Check if this file already exists for THIS hypothesis
        existing_filenames = [file["filename"] for file in existing_files]
        is_duplicate = file_name in existing_filenames

        if is_duplicate:
            st.warning(f"A file named '{file_name}' already exists for this hypothesis. Please choose a different file.")
        else:
            # Show upload button if not a duplicate
            if st.button("Upload File", key="upload_button"):
                print(f"DEBUG - Uploading file '{file_name}' for hypothesis ID '{hypothesis_id}'")
                # Read file content
                file_content = uploaded_file.read()

                # Save to GridFS
                file_id = fs.put(
                    file_content,
                    filename=file_name,
                    metadata={
                        "hypothesis_id": hypothesis_id,
                        "hypothesis_text": hypothesis_text,
                        "status": "unprocessed",  # Initial status
                        "upload_date": datetime.datetime.now().isoformat(),  # Use ISO format for consistent sorting
                    }
                )

                print(f"DEBUG - File successfully uploaded with ID: {file_id}")
                st.success(f"File '{file_name}' uploaded successfully!")
                
                # Add process file button
                if st.button("Process File", key="process_file_button"):
                    # Create a status message container
                    status_container = st.empty()
                    status_container.info("File processing initiated. This may take a moment...")
                    
                    # Update status to 'processing' in the database
                    db.fs.files.update_one(
                        {"_id": file_id},
                        {"$set": {"metadata.status": "processing"}}
                    )
                    
                    # Get values from session state
                    api_key = st.session_state.get("api_key", "")
                    provider = st.session_state.get("llm_provider", "openai")
                    model = st.session_state.get("llm_model", "gpt-4o")
                    
                    try:
                        # Call the process_file function
                        with st.spinner("Processing file... (this may take a minute or two)"):
                            result = process_file(db, fs, file_id, api_key, provider, model)
                        
                        if result["success"]:
                            # Show success message with expander for details
                            status_container.success("File processed successfully! Status: Ready for Analysis")
                                
                            with st.expander("View Processing Results"):
                                # Show metadata
                                st.subheader("File Metadata")
                                st.json(result.get("metadata", {}))
                                
                                # Show parsed data
                                st.subheader("Parsed Content")
                                st.json(result.get("parsed_data", {}))
                                
                                # If there's a confidence assessment, show it prominently
                                confidence_data = result.get("parsed_data", {}).get("confidence_assessment", {})
                                if confidence_data:
                                    confidence_score = confidence_data.get("confidence_score", 0)
                                    st.subheader("Confidence Assessment")
                                    
                                    # Display confidence score as a progress bar
                                    st.progress(float(confidence_score))
                                    st.write(f"**Score:** {confidence_score:.2f}")
                                    st.write(f"**Explanation:** {confidence_data.get('explanation', '')}")
                                    
                                    # Display strengths and limitations
                                    if "key_strengths" in confidence_data:
                                        st.write("**Key Strengths:**")
                                        for strength in confidence_data["key_strengths"]:
                                            st.write(f"- {strength}")
                                            
                                    if "key_limitations" in confidence_data:
                                        st.write("**Key Limitations:**")
                                        for limitation in confidence_data["key_limitations"]:
                                            st.write(f"- {limitation}")
                        else:
                            # Show error message
                            status_container.error(f"Error processing file: {result.get('error', 'Unknown error')}")
                            
                        # Refresh the page after a short delay
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        status_container.error(f"Error: {str(e)}")
                        
                        # Update file status to error in the database
                        db.fs.files.update_one(
                            {"_id": file_id},
                            {"$set": {
                                "metadata.status": "error",
                                "metadata.error_message": str(e)
                            }}
                        )

    # 4. NAVIGATION BUTTONS
    st.divider()
    
    col1, col2, spacer, col3 = st.columns([1, 1, 2, 1])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            return "back"
    
    with col3:
        # Check if there are any ready files before allowing to proceed
        if ready_for_analysis_files or processed_files:
            if st.button("Next →", use_container_width=True):
                return "next"
        else:
            st.button("Next →", use_container_width=True, disabled=True)
            st.caption("You need at least one processed file to continue")
    
    # Default return value if no navigation occurs
    return None