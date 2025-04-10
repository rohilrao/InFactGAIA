import streamlit as st
import datetime
import time
from file_processor import process_file
from evidence_recommender import display_evidence_recommendations
from bson.objectid import ObjectId

def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except:
            return id_value
    return id_value

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



    # NEW SECTION: Evidence Recommender
    # Get values from session state
    api_key = st.session_state.get("api_key", "")
    provider = st.session_state.get("provider", "openai")
    model = st.session_state.get("model", "gpt-4o")

    api_keys = {}        
    # Add the current provider's API key
    if api_key:
        api_keys[provider.lower()] = api_key
        
    # Try to get additional API keys from secrets
    if hasattr(st, 'secrets'):
        if 'SERPAPI_KEY' in st.secrets:
            api_keys['serpapi'] = st.secrets['SERPAPI_KEY']
        if 'SEMANTIC_SCHOLAR_API_KEY' in st.secrets:
            api_keys['semantic_scholar'] = st.secrets['SEMANTIC_SCHOLAR_API_KEY']

    # Display evidence recommendations with the correct provider settings
    evidence_selected, selected_evidence = display_evidence_recommendations(
        hypothesis_text,
        api_key=api_key,
        provider=provider,
        model=model,
        api_keys=api_keys
    )
        
    # If evidence was selected, offer to download it
    if evidence_selected and selected_evidence and 'pdf_url' in selected_evidence and selected_evidence['pdf_url']:
        st.info(f"**Recommended evidence found**: You can download the paper from the link above and upload it below.")
            
    st.divider()
    
    # 2. EXISTING FILES SECTION - Always fresh from database
    st.markdown("### :orange[Current Files]")

    # Get existing files for this hypothesis, excluding node state and rendered hypothesis files
    existing_files = list(db.fs.files.find({
        "metadata.hypothesis_id": hypothesis_id,
        "filename": {"$not": {"$regex": "node_state|rendered_hypothesis"}}
    }))

    # Categorize files by their status from the database
    unprocessed_files = [f for f in existing_files if f.get("metadata", {}).get("status") == "unprocessed"]
    processing_files = [f for f in existing_files if f.get("metadata", {}).get("status") == "processing"]
    ready_for_analysis_files = [f for f in existing_files if f.get("metadata", {}).get("status") == "ready_for_analysis" and 
                               f.get("metadata", {}).get("parsing_complete", False)]
    processed_files = [f for f in existing_files if f.get("metadata", {}).get("status") == "processed"]
    not_fit_files = [f for f in existing_files if f.get("metadata", {}).get("status") == "not fit for analysis"]
    error_files = [f for f in existing_files if f.get("metadata", {}).get("status") == "error"]
    pending_validation_files = [f for f in existing_files if f.get("metadata", {}).get("status") == "pending_validation"]
    
    # Any files that are ready for analysis or already processed
    analyzed_files = ready_for_analysis_files #+ processed_files
    
    # If we have any pending validation files, we should show a processing indicator
    if pending_validation_files:
        st.info("⏳ Some files are being validated. Please wait for validation to complete.")
        st.progress(0.7, "Validation in progress")
        
        # Add automatic refresh capability for pending files
        if st.button("Refresh Status", key="refresh_validation", use_container_width=True):
            st.rerun()
    
    # Any files that are ready for analysis or already processed
    analyzed_files = ready_for_analysis_files 

    if existing_files:
        st.caption(f"{len(existing_files)} file(s) associated with this hypothesis")

        # Create a scrollable container
        file_container = st.container(height=250, border=True)

        # Display files within the scrollable container
        with file_container:
            for idx, file in enumerate(existing_files):
                file_id = file["_id"]
                filename = file["filename"]
                
                # Get status from metadata (more reliable)
                status = file.get("metadata", {}).get("status", "unknown")

                # Create a row for each file
                col1, col2, col3, col4 = st.columns([3, 3, 1, 1])

                # File name column
                with col1:
                    st.write(f"**{filename}**")
                    st.caption(f"ID: {str(file_id)[-6:]}")

                # Status column
                with col2:
                    if status == "ready_for_analysis":
                        st.markdown("<span style='color:green'>🔍 Ready for Analysis</span>", unsafe_allow_html=True)
                    elif status == "processing":
                        st.markdown("<span style='color:orange'>⏳ Processing</span>", unsafe_allow_html=True)
                    elif status == "processed":
                        st.markdown("<span style='color:green'>✅ Processed</span>", unsafe_allow_html=True)
                    elif status == "not fit for analysis":
                        st.markdown("<span style='color:red'>❌ Not Fit for Analysis</span>", unsafe_allow_html=True)
                    elif status == "pending_validation":
                        st.markdown("<span style='color:blue'>🔄 Validating Data</span>", unsafe_allow_html=True)
                    elif status == "error":
                        st.markdown("<span style='color:red'>⚠️ Error</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:#888'>⚪ Unprocessed</span>", unsafe_allow_html=True)
                        
                # Add expander for error messages when status is "error"
                if status == "error":
                    with st.expander("See error details"):
                        error_message = file.get("metadata", {}).get("error_message", "Unknown error")
                        st.error(f"**Error:** {error_message}")
                        st.warning("Please try uploading a different file or contact the developer if the issue persists.")

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
                    if status != "processed":
                        if st.button("Delete", key=f"delete_{idx}", use_container_width=True):
                            if delete_file(db, fs, file_id):
                                st.rerun()

                # Add a separator between files
                if idx < len(existing_files) - 1:
                    st.divider()
    else:
        st.info("No files uploaded yet. Upload your first file below.")

    st.divider()
    
    # 3. FILE WORKFLOW SECTION - Determine what to show based on existing file statuses
    
    # CASE 1: Analyzed files exist - prompt user to proceed to next step
    if analyzed_files:
        st.success("✅ You have files ready for analysis. Please proceed to the next step.")
        
        # Get the most recent ready file
        analyzed_files.sort(key=lambda x: x.get("metadata", {}).get("last_updated", 
                           x.get("metadata", {}).get("upload_date", "")), reverse=True)
        recent_file = analyzed_files[0]
        
        # Store this file ID in session state for future steps
        st.session_state["analyzed_file_id"] = str(recent_file["_id"])
        
        # Display a Next button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Next →", key="next_with_analyzed_files", use_container_width=True):
                return "next"
    
    # CASE 2: Processing files exist - show processing indicator
    elif processing_files:
        st.info("⏳ Processing file(s). Please wait until processing completes.")
        st.progress(0.5, "Processing in progress")
        
        # Add automatic refresh capability
        if st.button("Refresh Status", use_container_width=True):
            st.rerun()
    
    # CASE 3: Unprocessed files exist - prompt user to process them
    elif unprocessed_files:
        st.warning(f"You have {len(unprocessed_files)} unprocessed file(s). Please process them before uploading new files.")
        
        # Get the first unprocessed file
        file_to_process = unprocessed_files[0]
        file_id = file_to_process["_id"]
        filename = file_to_process["filename"]
        
        # Show process button
        if st.button("Process File", key="process_existing_file", use_container_width=True):
            # Update UI to show processing state
            status_container = st.empty()
            status_container.info(f"Processing file '{filename}'... This may take a moment...")
            
            # Update status to 'processing' in the database
            db.fs.files.update_one(
                {"_id": file_id},
                {"$set": {"metadata.status": "processing"}}
            )
            
            # Get values from session state
            api_key = st.session_state.get("api_key",)
            provider = st.session_state.get("provider")
            model = st.session_state.get("model")
            
            try:
                # Call the process_file function
                with st.spinner(f"Processing '{filename}'... (this may take a minute or two)"):
                    result = process_file(db, fs, file_id, api_key, provider, model)
                
                if result["success"]:
                    # Update the file document in the database with the parsed data
                    # Don't need to update here as process_file already updates the database
                    # Just display the results from the process_file response
                    
                    # Update success message
                    status_container.success(f"File '{filename}' processed successfully!")
                    
                    # Display results
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
                    
                    # Update file status back to unprocessed
                    db.fs.files.update_one(
                        {"_id": file_id},
                        {"$set": {"metadata.status": "unprocessed"}}
                    )
                
                # Refresh the page after a short delay
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
                # Update file status to error in the database
                db.fs.files.update_one(
                    {"_id": file_id},
                    {"$set": {
                        "metadata.status": "unprocessed",
                        "metadata.error_message": str(e)
                    }}
                )
                
                time.sleep(1)
                st.rerun()
    
    # CASE 4: No files exist - show upload form
    else:
        st.markdown("### :orange[Upload New File]")
        st.write("You can upload one file at a time. Please wait for processing to complete before uploading another file.")
        
        # File uploader
        uploaded_file = st.file_uploader("Select a file to upload", type=["txt", "pdf", "png", "jpg", "html", "csv"])

        # Logic for handling file upload
        if uploaded_file:
            file_name = uploaded_file.name
            
            # Check if this file already exists for THIS hypothesis - check based on DB
            existing_filenames = [file["filename"] for file in existing_files]
            is_duplicate = file_name in existing_filenames

            if is_duplicate:
                st.warning(f"⚠️ A file named '{file_name}' already exists for this hypothesis. Please choose a different file.")
            else:
                # Show upload button if file is not a duplicate
                if st.button("Upload File", key="upload_button", use_container_width=True):
                    # Read file content
                    file_content = uploaded_file.read()

                    # Save to GridFS with proper metadata
                    file_id = fs.put(
                        file_content,
                        filename=file_name,
                        metadata={
                            "hypothesis_id": hypothesis_id,
                            "hypothesis_text": hypothesis_text,
                            "status": "unprocessed",  # Initial status
                            "upload_date": datetime.datetime.now().isoformat(),
                        }
                    )

                    st.success(f"✅ File '{file_name}' uploaded successfully!")
                    
                    # Refresh to show the new file in the list
                    time.sleep(0.5)
                    st.rerun()
    
    # 4. DISPLAY PARSED DATA SECTION (only if we have analyzed files)
    
    if analyzed_files or not_fit_files:
        st.divider()
        st.markdown("### :orange[Parsed Data Preview]")
        
        # Create dropdown with analyzed files and not fit files
        file_options = {f"{file['filename']} (ID: {str(file['_id'])[-6:]})": str(file["_id"]) 
                        for file in analyzed_files + not_fit_files}
        
        if file_options:
            selected_file = st.selectbox(
                "Select a file to view its parsed data:", 
                options=list(file_options.keys()),
                key="parsed_data_file_selector"
            )
            
            if selected_file:
                selected_file_id = file_options[selected_file]
                
                # Convert to ObjectId if needed
                object_id = ensure_object_id(selected_file_id)
                
                # Query with the proper ID
                file_doc = db.fs.files.find_one({"_id": object_id})
                
                if file_doc and "metadata" in file_doc:
                    file_status = file_doc.get("metadata", {}).get("status", "")
                    
                    # First display file status prominently
                    if file_status == "ready_for_analysis":
                        st.success(f"✅ This file is **ready for analysis**")
                    elif file_status == "not fit for analysis":
                        st.error(f"❌ This file is **not fit for Bayesian analysis**")
                    
                    # Get parsed data if available
                    if "parsed_data" in file_doc["metadata"]:
                        parsed_data = file_doc["metadata"]["parsed_data"]
                        
                        # Display the parsed data in a nice format
                        st.markdown(f"#### Data from: {file_doc['filename']}")
                        
                        # Create tabs for different views - ONLY for parsed data
                        summary_tab, raw_data_tab = st.tabs(["Summary View", "Raw JSON Data"])
                        
                        with summary_tab:
                            # Display confidence assessment if available
                            confidence_data = parsed_data.get("confidence_assessment", {})
                            if confidence_data:
                                confidence_score = confidence_data.get("confidence_score", 0)
                                
                                # Create columns for better layout
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.subheader("Confidence")
                                    # Display confidence score as a progress bar
                                    st.progress(float(confidence_score))
                                    # Determine color based on score
                                    if confidence_score >= 0.75:
                                        score_color = "green"
                                    elif confidence_score >= 0.5:
                                        score_color = "orange"
                                    else:
                                        score_color = "red"
                                    
                                    st.markdown(f"<h3 style='color:{score_color}'>{confidence_score:.2f}</h3>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.subheader("Assessment")
                                    st.write(f"{confidence_data.get('explanation', '')}")
                                    
                                    # Display strengths and limitations in expandable sections
                                    if "key_strengths" in confidence_data:
                                        with st.expander("Key Strengths"):
                                            for strength in confidence_data["key_strengths"]:
                                                st.write(f"- {strength}")
                                            
                                    if "key_limitations" in confidence_data:
                                        with st.expander("Key Limitations"):
                                            for limitation in confidence_data["key_limitations"]:
                                                st.write(f"- {limitation}")
                            
                            # Display other parsed data components
                            if "numerical_values" in parsed_data:
                                with st.expander("Numerical Values"):
                                    st.write(parsed_data["numerical_values"])
                            
                            if "metadata" in parsed_data:
                                with st.expander("Metadata"):
                                    st.json(parsed_data["metadata"])
                            
                            if "issues" in parsed_data:
                                with st.expander("Issues"):
                                    for issue in parsed_data["issues"]:
                                        st.warning(issue)
                        
                        with raw_data_tab:
                            st.caption("Complete data structure returned by the parser")
                            st.json(parsed_data)
                        
                        # AFTER the tabs, display validation results separately for "not fit for analysis" files
                        if file_status == "not fit for analysis":
                            st.divider()
                            validation_result = file_doc.get("metadata", {}).get("validation_result", {})
                            
                            st.markdown("### :red[Unsuitable file]")
                            
                            # Display reason with detailed explanation
                            st.error(f"**Either the document is unsuitable or there was a failure to extract numerical data that could be relevant to the hypothesis. If you think this is a mistake then please contact the developer.**")
                            st.write(f"**Reason:** {validation_result.get('reason', 'The numerical values in this file are not appropriate for the hypothesis.')}")
                            
                            # Display suggestions with guidance
                            if "suggestions" in validation_result and validation_result["suggestions"]:
                                st.info("**Recommendations:**")
                                st.write(validation_result["suggestions"])
                            
                            # Display key points about the numerical values
                            if "key_points" in validation_result and validation_result["key_points"]:
                                st.subheader("Key Points")
                                for point in validation_result["key_points"]:
                                    st.markdown(f"- {point}")
                    
                    else:
                        st.warning("No parsed data available for this file.")
                else:
                    st.warning("File metadata not found.")

    # 5. NAVIGATION BUTTONS
    st.divider()
    
    col1, col2, spacer, col3 = st.columns([1, 1, 2, 1])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            return "back"
    
    with col3:
        # Only enable Next button if we have analyzed files
        if analyzed_files:
            if st.button("Next →", use_container_width=True):
                return "next"
        else:
            st.button("Next →", use_container_width=True, disabled=True)
            st.caption("You need at least one \"ready for analysis\" file to continue")
    
    # Default return value if no navigation occurs
    return None