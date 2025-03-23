import streamlit as st
import os
import tempfile
import datetime
import sys
import math
import pandas as pd
from bson.objectid import ObjectId

def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except:
            return id_value
    return id_value

class StreamToLogger:
    """Helper class to redirect stdout/stderr to Streamlit UI."""
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.logs = []
        
    def write(self, text):
        self.logs.append(text)
        self.placeholder.text('\n'.join(self.logs))
        
    def flush(self):
        pass

def display_evidence_step(db, fs, hypothesis_collection):
    """
    Handles Step 6: Evidence Processing
    
    Args:
        db: MongoDB database connection
        fs: GridFS instance
        hypothesis_collection: MongoDB collection for hypotheses
        
    Returns:
        str: Navigation action - "back", "next", or None
    """
    st.header("Step 6: Evidence Processing")
    
    # Get hypothesis information
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
        st.stop()

    # Get the current hypothesis text and file info from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
    st.write(f"**Hypothesis:** {hypothesis_text}")
    st.markdown("---")
    
    # Check if we have a file ready for processing
    if not st.session_state.get("file_ready_for_processing", False):
        st.warning("No file ready for processing. Please complete Step 5 first.")
        
        if st.button("← Back to Step 5"):
            return "back"
        st.stop()
    
    # Get essential data from session state
    file_id = st.session_state.get("current_file_id")
    filename = st.session_state.get("current_filename")
    parsed_data = st.session_state.get("parsed_data")
    validated_code = st.session_state.get("validated_code")
    l_plus = st.session_state.get("l_plus")
    l_minus = st.session_state.get("l_minus")
    node = st.session_state.get("node")
    
    if not node or not validated_code:
        st.error("Required session data is missing. Please go back to Step 5.")
        if st.button("← Return to Step 5"):
            return "back"
        st.stop()
    
    st.subheader(f"Processing file: {filename}")
    
    # Show a summary of the analysis so far
    st.write("### Analysis Summary")
    
    # Convert log odds to probability for display
    p_h_given_d = 1 / (1 + math.exp(-l_plus + l_minus))
    
    st.write(f"**l_plus (log P(data | hypothesis)):** {l_plus:.4f}")
    st.write(f"**l_minus (log P(data | not hypothesis)):** {l_minus:.4f}")
    st.write(f"**Probability of hypothesis given this data:** {p_h_given_d:.2%}")
    
    # Import InFactRenderer
    from InFactRenderer import InFactRenderer
    
    # Check if we've already processed the file
    if st.session_state.get("file_processed", False):
        st.success("File processing complete!")
        st.write("### Final Node State and Analysis")
        
        # Display node state visualization
        if node:
            # Get current posterior and convert to probability
            posterior_prob = 1 / (1 + math.exp(-node.current_posterior))
            
            # Display current probability
            st.write(f"**Current probability of hypothesis:** {posterior_prob:.2%}")
            
            # Display confidence interval if available
            lower, upper = node._calculate_uncertainty()
            st.write(f"**95% Confidence Interval:** ({lower:.2%}, {upper:.2%})")
            
            # Create a progress bar to visualize probability
            st.progress(posterior_prob)
            
            # Display data points table
            if node.data_points:
                st.write("### Evidence Summary")
                
                # Create a DataFrame for display
                data_rows = []
                for i, dp in enumerate(node.data_points):
                    metadata = dp.get('metadata', {})
                    filename = metadata.get('filename', f"Data point {i+1}")
                    
                    # Calculate Bayes factor
                    bayes_factor = math.exp(dp['l_plus'] - dp['l_minus'])
                    
                    # Map strength of evidence
                    if bayes_factor < 1:
                        evidence_strength = "Evidence against hypothesis"
                        evidence_color = "red"
                    elif bayes_factor < 3:
                        evidence_strength = "Weak evidence"
                        evidence_color = "orange"
                    elif bayes_factor < 10:
                        evidence_strength = "Moderate evidence"
                        evidence_color = "blue"
                    elif bayes_factor < 30:
                        evidence_strength = "Strong evidence"
                        evidence_color = "green"
                    else:
                        evidence_strength = "Very strong evidence"
                        evidence_color = "darkgreen"
                    
                    data_rows.append({
                        "File": filename,
                        "l_plus": round(dp['l_plus'], 4),
                        "l_minus": round(dp['l_minus'], 4),
                        "Bayes Factor": round(bayes_factor, 2),
                        "Evidence Strength": evidence_strength
                    })
                
                df = pd.DataFrame(data_rows)
                st.dataframe(df)
        
        # Option to download node state
        if "node_state_file_id" in st.session_state:
            node_state_file_id = st.session_state["node_state_file_id"]
            if fs.exists(node_state_file_id):
                st.download_button(
                    label="⬇️ Download Node State",
                    data=fs.get(node_state_file_id).read(),
                    file_name=f"{filename}_node_state.json",
                    mime="application/json"
                )
        
        # Option to download analysis
        if "analysis_file_id" in st.session_state:
            analysis_file_id = st.session_state["analysis_file_id"]
            if fs.exists(analysis_file_id):
                file_content = fs.get(analysis_file_id).read().decode()
                st.download_button(
                    label="⬇️ Download Analysis Report",
                    data=file_content,
                    file_name=f"{filename}_analysis.html",
                    mime="text/html"
                )
                st.components.v1.html(file_content, height=600, scrolling=True)
    
    else:
        # Display the validated code in a read-only text area
        with st.expander("Review Finalized Analysis Code"):
            st.code(validated_code, language="python")
        
        # Offer to process the file
        st.subheader("Process File")
        st.write("Click below to process the file using the validated analysis code.")
        
        process_col1, process_col2 = st.columns([1, 1])
        
        with process_col1:
            if st.button("🚀 Process Evidence File"):
                log_placeholder = st.empty()  # Create placeholder for logs
                output_buffer = StreamToLogger(log_placeholder)
                sys.stdout, sys.stderr = output_buffer, output_buffer  # Redirect logs to UI
                
                try:
                    with st.spinner("Processing file..."):
                        # Fetch file content and create temporary file
                        file_obj = fs.get(ensure_object_id(file_id))
                        temp_dir = tempfile.gettempdir()
                        temp_file_path = os.path.join(temp_dir, filename)
                        
                        with open(temp_file_path, "wb") as f:
                            f.write(file_obj.read())
                        
                        # Process the data interactively with validated code
                        new_posterior, (lower, upper) = node.process_data_interactively(temp_file_path, validated_code)
                        
                        # Save node state to a temporary file
                        temp_node_state_path = os.path.join(temp_dir, f"node_state_{filename}.json")
                        node.save(temp_node_state_path)
                        
                        # Upload node state to GridFS
                        with open(temp_node_state_path, "r", encoding="utf-8") as f:
                            node_state_content = f.read()
                        
                        state_file_id = fs.put(
                            node_state_content.encode(),
                            filename=f"node_state_{filename}.json",
                            content_type="application/json"
                        )
                        
                        # Render analysis
                        temp_analysis_path = os.path.join(temp_dir, f"analysis_{filename}.html")
                        renderer = InFactRenderer()
                        renderer.render_analysis(node, temp_analysis_path)
                        
                        # Upload analysis to GridFS
                        with open(temp_analysis_path, "r", encoding="utf-8") as f:
                            analysis_content = f.read()
                        
                        analysis_file_id = fs.put(
                            analysis_content.encode(),
                            filename=f"analysis_{filename}.html",
                            content_type="text/html"
                        )
                        
                        # Update file status in database
                        db.fs.files.update_one(
                            {"_id": ensure_object_id(file_id)},
                            {
                                "$set": {
                                    "status": "processed",
                                    "node_state_file_id": state_file_id,
                                    "last_processed_at": datetime.datetime.utcnow(),
                                    "analysis_file_id": analysis_file_id,
                                }
                            }
                        )
                        
                        # Store IDs in session state
                        st.session_state["node_state_file_id"] = state_file_id
                        st.session_state["analysis_file_id"] = analysis_file_id
                        st.session_state["file_processed"] = True
                        
                        # Cleanup temporary files
                        try:
                            os.remove(temp_file_path)
                            os.remove(temp_node_state_path)
                            os.remove(temp_analysis_path)
                        except Exception as e:
                            st.warning(f"Failed to remove temporary files: {str(e)}")
                        
                        st.success("File processed successfully!")
                        return "reload"
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                
                finally:
                    # Restore stdout/stderr
                    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        
        with process_col2:
            if st.button("Edit Analysis Code"):
                # Go back to code editing in step 5
                st.session_state["file_ready_for_processing"] = False
                return "back"
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back to Step 5"):
            # Keep most session state but reset processing flag
            st.session_state["file_ready_for_processing"] = False
            return "back"
    
    with col2:
        if st.button("Process New Evidence"):
            # Clean up session state for new file
            for key in ["current_file_id", "current_filename", "parsed_data", 
                       "generated_code", "validated_code", "l_plus", "l_minus",
                       "file_ready_for_processing", "file_processed",
                       "node_state_file_id", "analysis_file_id"]:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Keep the node object to maintain state
            return "back_to_step5"
    
    with col3:
        if st.button("View All Results"):
            # Navigate to a hypothetical results page
            return "next"
            
    return None  # No action taken