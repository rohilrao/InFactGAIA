import streamlit as st
import os
import json
import math
from pathlib import Path
from bson.objectid import ObjectId
import tempfile

# Import the InFactRenderer
from InFact.utils.InFactRenderer import InFactRenderer

def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except Exception:
            pass  # Fall through to the return below
    # Return the original value if conversion failed or wasn't needed
    return id_value

def json_serialize_with_datetime(obj):
    """Custom JSON serializer that handles datetime objects."""
    from datetime import datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def node_state_to_json(node):
    """Convert node state to JSON string with datetime handling."""
    data = {
        'hypothesis': node.hypothesis,
        'prior_log_odds': node.prior_log_odds,
        'current_posterior': node.current_posterior,
        'provider_info': {
            'type': node.llm_provider.__class__.__name__,
            'model': node.llm_provider.model,
        },
        'data_points': [
            {
                'metadata': dp['metadata'],
                'raw_data': dp['raw_data'],
                'l_plus': dp['l_plus'],
                'l_minus': dp['l_minus'],
                'posterior': dp['posterior'],
                'confidence_assessment': dp.get('confidence_assessment', {}),
                'analysis_rationale': dp.get('analysis_rationale', '')
            }
            for dp in node.data_points
        ]
    }
    
    return json.dumps(data, indent=2, default=json_serialize_with_datetime)

def display_results_step(db, fs, hypothesis_collection):
    """
    Handles Step 5: Process Results and Visualization
    
    Args:
        db: MongoDB database connection
        fs: GridFS instance
        hypothesis_collection: MongoDB collection for hypotheses
        
    Returns:
        str: Navigation action - "back", "next", or None
    """
    # Add styling
    st.markdown("""
    <style>
    .results-container {
        background-color: rgba(0, 100, 0, 0.1);
        padding: 15px; 
        border-radius: 5px; 
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .nav-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .hypothesis-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .rendered-html {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        margin-top: 20px;
        height: 500px;
        overflow: auto;
    }
    .download-buttons {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### :orange[Process Results]")
    
    # Get hypothesis information
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
        if st.button("← Back to Hypothesis Setup"):
            return "back"
        st.stop()

    # Get the current hypothesis text from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        if st.button("← Back to Hypothesis Setup"):
            return "back"
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    hypothesis_description = hypothesis_entry.get("short_description", "No description provided")
    
    # Display hypothesis info
    st.write("Hypothesis Information:")
    st.write("**Hypothesis ID:**", hypothesis_id)   
    st.write(f"**Hypothesis:** {hypothesis_text}")
    st.markdown("---")
    
    # Get InFactNode and file information from session
    node = st.session_state.get("infact_node", None)
    file_id = st.session_state.get("current_file_id", None)
    new_posterior = st.session_state.get("new_posterior", None)
    
    if not node:
        st.error("InFactNode not found in session state. Please go back to initialize the analysis environment.")
        if st.button("← Back to Analysis Setup"):
            return "back"
        st.stop()
        
    if not file_id or not new_posterior:
        st.error("File information or analysis results not found. Please complete the code analysis step first.")
        if st.button("← Back to Code Review"):
            return "back"
        st.stop()
    
    # Get file information
    file_obj = db.fs.files.find_one({"_id": ensure_object_id(file_id)})
    if not file_obj:
        st.error("Could not find file data in database.")
        if st.button("← Back to File Upload"):
            return "back"
        st.stop()
        
    filename = file_obj["filename"]
    file_path = os.path.basename(filename)
    
    # Get validated code and results from the database
    analysis_code = file_obj.get("analysis_code", "No analysis code found")
    analysis_results = file_obj.get("analysis_results", {})
    parsed_data = file_obj.get("parsed_data", {})
    
    l_plus = analysis_results.get("l_plus", 0)
    l_minus = analysis_results.get("l_minus", 0)
    
    # Process results section
    st.markdown("## Processing Results")
    
    with st.spinner("Processing data and updating belief..."):
        try:
            # Create a basic metadata structure since we don't have extract_metadata
            metadata = {
                "filename": file_path,
                "file_id": str(file_id),
                "processed_date": file_obj.get("uploadDate", "Unknown date")
            }
            
            # Update node's current_posterior if not already done
            node.current_posterior = new_posterior
            
            # Create a data point structure similar to what's created in process_data
            data_point = {
                'raw_data': parsed_data,
                'metadata': metadata,
                'l_plus': l_plus,
                'l_minus': l_minus,
                'posterior': new_posterior,
                'confidence_assessment': parsed_data.get('confidence_assessment', {
                    'confidence_score': 0.5,  # Default moderate confidence
                    'explanation': 'Automatically processed data point',
                    'key_strengths': ['Systematic analysis'],
                    'key_limitations': ['Limited context awareness']
                }),
                'analysis_rationale': analysis_code
            }
            
            # Add data point to node if not already present
            # Check if this data point is already in the node to avoid duplicates
            file_ids_in_node = [dp.get('metadata', {}).get('file_id', "") for dp in node.data_points]
            if str(file_id) not in file_ids_in_node:
                node.data_points.append(data_point)
                st.success(f"Added data point for file: {file_path}")
            else:
                st.info(f"Data point for file {file_path} already exists in the node")
            
            # Calculate probability and uncertainty
            probability = node._to_probability(new_posterior)
            lower, upper = node._calculate_uncertainty()
            
            # === Handle node state and HTML rendering ===
            
            # Create the node state JSON directly
            node_state_content = node_state_to_json(node)
            st.session_state["node_state_json"] = node_state_content
            
            # Check if we already have a node state for this hypothesis
            existing_node_state = db.fs.files.find_one({
                "metadata.type": "node_state",
                "metadata.hypothesis_id": str(hypothesis_id),
                "metadata.is_latest": True
            })
            
            # Mark any previous "latest" node states as not latest
            if existing_node_state:
                db.fs.files.update_many(
                    {"metadata.type": "node_state", "metadata.hypothesis_id": str(hypothesis_id)},
                    {"$set": {"metadata.is_latest": False}}
                )
            
            # Define the metadata for hypothesis state
            hypothesis_state_metadata = {
                "type": "node_state",
                "hypothesis_id": str(hypothesis_id),
                "file_id": str(file_id),  # The file that triggered this update
                "is_latest": True
            }
            
            # Upload the node state to GridFS for the hypothesis
            hypothesis_state_id = fs.put(
                node_state_content.encode(),
                filename=f"node_state_hypothesis_{hypothesis_id}.json",
                content_type="application/json",
                metadata=hypothesis_state_metadata
            )
            
            # Define metadata for file-specific state
            file_state_metadata = {
                "type": "node_state",
                "hypothesis_id": str(hypothesis_id),
                "file_id": str(file_id),
                "is_latest": False  # This is a file-specific snapshot
            }
            
            # Upload a file-specific state
            file_state_id = fs.put(
                node_state_content.encode(),
                filename=f"node_state_file_{file_id}.json",
                content_type="application/json",
                metadata=file_state_metadata
            )
            
            # Update the file record to reference this node state
            db.fs.files.update_one(
                {"_id": ensure_object_id(file_id)},
                {"$set": {"node_state_file_id": str(file_state_id)}}
            )
            
            # Now handle the HTML rendering with a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Render HTML visualization
                renderer = InFactRenderer()
                html_output = renderer.render_analysis(node)
                
                # Store HTML in session state for display and download
                st.session_state["html_output"] = html_output
                
                # Save HTML to a temporary file
                html_temp_path = os.path.join(temp_dir, f"hypothesis_{hypothesis_id}.html")
                with open(html_temp_path, 'w', encoding='utf-8') as f:
                    f.write(html_output)
                
                # Define HTML metadata
                html_metadata = {
                    "type": "rendered_html",
                    "hypothesis_id": str(hypothesis_id),
                    "file_id": str(file_id),
                    "is_latest": True
                }
                
                # Mark any previous "latest" HTML as not latest
                db.fs.files.update_many(
                    {"metadata.type": "rendered_html", "metadata.hypothesis_id": str(hypothesis_id)},
                    {"$set": {"metadata.is_latest": False}}
                )
                
                # Upload new HTML
                with open(html_temp_path, 'rb') as f:
                    html_file_id = fs.put(
                        f,
                        filename=f"rendered_hypothesis_{hypothesis_id}.html",
                        content_type="text/html",
                        metadata=html_metadata
                    )
            
            # Update the hypothesis collection with latest information
            hypothesis_collection.update_one(
                {"_id": hypothesis_id},
                {"$set": {
                    "latest_node_state_id": str(hypothesis_state_id),
                    "latest_html_id": str(html_file_id),
                    "latest_file_processed": str(file_id),
                    "current_posterior": new_posterior,
                    "probability": probability,
                    "confidence_interval": [lower, upper],
                    "last_updated": file_obj.get("uploadDate", "Unknown date")
                }}
            )
            
            # Display results
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.write("### Analysis Results")
            st.write(f"**l_plus (log P(data | hypothesis)):** {l_plus:.4f}")
            st.write(f"**l_minus (log P(data | not hypothesis)):** {l_minus:.4f}")
            st.write(f"**New posterior log odds:** {new_posterior:.4f}")
            st.write(f"**Probability of hypothesis:** {probability:.2%}")
            st.write(f"**Confidence interval (95%):** ({lower:.2%}, {upper:.2%})")
            st.write("---")
            st.write(f"**Node state saved to GridFS with ID:** `{hypothesis_state_id}`")
            st.write(f"**HTML visualization saved to GridFS with ID:** `{html_file_id}`")
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing results: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
    
    # Display the rendered HTML
    st.markdown("## Rendered Visualization")
    
    # Create tabs for viewing HTML and raw node data
    tabs = st.tabs(["Visualization", "Raw Node Data", "Download Files"])
    
    with tabs[0]:
        # Display the HTML in an iframe if we have it in session state
        html_output = st.session_state.get("html_output", None)
        if html_output:
            # Use base64 encoding for the HTML content
            import base64
            encoded_html = base64.b64encode(html_output.encode()).decode()
            
            # Display in an iframe
            st.markdown(f'<div class="rendered-html"><iframe src="data:text/html;base64,{encoded_html}" width="100%" height="100%"></iframe></div>', unsafe_allow_html=True)
        else:
            st.warning("HTML visualization not available in session state")
    
    with tabs[1]:
        # Display raw node data as JSON
        node_data = {
            'hypothesis': node.hypothesis,
            'prior_log_odds': node.prior_log_odds,
            'current_posterior': node.current_posterior,
            'probability': probability,
            'confidence_interval': [lower, upper],
            'data_points_count': len(node.data_points),
            'latest_data_point': {
                'file': file_path,
                'l_plus': l_plus,
                'l_minus': l_minus,
                'posterior': new_posterior
            } if str(file_id) not in file_ids_in_node else "Already in node"
        }
        st.json(node_data)
    
    with tabs[2]:
        st.markdown("### Download Files")
        st.write("Download the node state and visualization for offline use or further analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download button for node state JSON
            if "node_state_json" in st.session_state:
                st.download_button(
                    label="Download Node State (JSON)",
                    data=st.session_state["node_state_json"],
                    file_name=f"hypothesis_{hypothesis_id}_state.json",
                    mime="application/json",
                    key="download_json"
                )
            else:
                st.warning("Node state JSON not available for download")
        
        with col2:
            # Download button for HTML
            if "html_output" in st.session_state:
                st.download_button(
                    label="Download Visualization (HTML)",
                    data=st.session_state["html_output"],
                    file_name=f"hypothesis_{hypothesis_id}_visualization.html",
                    mime="text/html",
                    key="download_html"
                )
            else:
                st.warning("HTML visualization not available for download")
    
    # Navigation buttons
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("← Back to Code Review"):
            return "back"
    
    with col2:
        if st.button("Process Another File", type="primary"):
            # Clear current file-specific session data but keep the node
            for key in ["current_file_id", "current_filename", "parsed_data", 
                      "generated_code", "current_code", "validated_code", 
                      "l_plus", "l_minus", "edit_mode", "new_posterior",
                      "simple_analysis", "tech_analysis"]:
                if key in st.session_state:
                    del st.session_state[key]
            return "restart_file_processing"
    
    return None  # No action taken