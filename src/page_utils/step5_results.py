import streamlit as st
import os
import tempfile
from bson.objectid import ObjectId
import math
from datetime import datetime
import base64
from MongoInFactRenderer import MongoInFactRenderer
from infact_utils import call_llm
from bayesian_analysis_utils import _to_probability, _calculate_uncertainty

def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except Exception:
            pass  # Fall through to the return below
    # Return the original value if conversion failed or wasn't needed
    return id_value

def display_results_step(db, fs, hypothesis_collection):
    """
    Handles Step 5: Results and Visualization
    
    Args:
        db: MongoDB database connection
        fs: GridFS instance
        hypothesis_collection: MongoDB collection for hypotheses
        
    Returns:
        str: Navigation action - "back", "restart", or None
    """
    # Add styling for results page
    st.markdown("""
    <style>
    .results-container {
        background-color: rgba(0, 100, 0, 0.1);
        padding: 15px; 
        border-radius: 5px; 
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .evidence-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .evidence-header {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px 5px 0 0;
        margin: -15px -15px 15px -15px;
        border-bottom: 1px solid #e0e0e0;
    }
    .navigation-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .confidence-high {
        color: #047857;
        font-weight: bold;
    }
    .confidence-medium {
        color: #b45309;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc2626;
        font-weight: bold;
    }
    .probability-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2b6cb0;
        text-align: center;
        margin: 10px 0;
    }
    .confidence-interval {
        text-align: center;
        color: #4a5568;
        margin-bottom: 20px;
    }
    .interpretation {
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
        font-size: 1.2rem;
    }
    .iframe-container {
        width: 100%;
        height: 800px;
        border: none;
        overflow: hidden;
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
    
    # Get the current hypothesis entry from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        if st.button("← Back to Hypothesis Setup"):
            return "back"
        st.stop()

    # Extract hypothesis details
    hypothesis_text = hypothesis_entry["text"]
    hypothesis_description = hypothesis_entry.get("description", "No description provided")
    
    # Check if the hypothesis has any data points
    if "node_metadata" not in hypothesis_entry or "data_points" not in hypothesis_entry["node_metadata"] or not hypothesis_entry["node_metadata"]["data_points"]:
        st.warning("No analysis has been performed on this hypothesis yet. Please go back to Step 4 to analyze data.")
        if st.button("← Back to Code Review"):
            return "back"
        st.stop()
    
    # Display hypothesis info
    st.write("**Hypothesis ID:**", hypothesis_id)
    st.write(f"**Hypothesis:** {hypothesis_text}")
    if hypothesis_description != "No description provided":
        st.write(f"**Description:** {hypothesis_description}")

    st.markdown("---")
    
    # Get current posterior and data points
    data_points = hypothesis_entry["node_metadata"]["data_points"]
    current_posterior = hypothesis_entry["node_metadata"].get("current_posterior", 0.0)
    
    # Calculate current probability and uncertainty
    probability = _to_probability(current_posterior)
    lower, upper = _calculate_uncertainty(current_posterior, data_points)
    
    # Display current assessment
    st.subheader("Current Assessment")
    
    # Determine interpretation based on probability
    interpretation = ""
    if probability > 0.99:
        interpretation = "Virtually Certain"
    elif probability > 0.95:
        interpretation = "Extremely Likely"
    elif probability > 0.90:
        interpretation = "Very Likely"
    elif probability > 0.66:
        interpretation = "Likely"
    elif probability > 0.33:
        interpretation = "Uncertain"
    elif probability > 0.10:
        interpretation = "Unlikely"
    elif probability > 0.05:
        interpretation = "Very Unlikely"
    elif probability > 0.01:
        interpretation = "Extremely Unlikely"
    else:
        interpretation = "Virtually Impossible"
    
    st.markdown(f'<div class="probability-value">{probability:.1%}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="confidence-interval">95% Confidence Interval: ({lower:.1%}, {upper:.1%})</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="interpretation">{interpretation}</div>', unsafe_allow_html=True)
    
    # Display tabs for different views
    results_tab, evidence_tab, visual_tab = st.tabs(["Summary", "Evidence Details", "Visualization"])
    
    with results_tab:
        # Display summary of the analysis
        st.subheader("Analysis Summary")
        
        # Add a bit more space
        st.write("")
        
        # Create summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points Analyzed", len(data_points))
        with col2:
            # Format the posterior for display
            posterior_display = f"{current_posterior:.2f}"
            st.metric("Log Odds", posterior_display)
        with col3:
            # For last evidence, calculate the change
            if len(data_points) > 0:
                last_point = data_points[-1]
                change = last_point["l_plus"] - last_point["l_minus"]
                change_display = f"{change:+.2f}"
                st.metric("Last Evidence Impact", change_display)
        
        # Add a list of the files analyzed
        st.write("##### Files Analyzed")
        for idx, point in enumerate(data_points, 1):
            filename = point.get("filename", f"File {idx}")
            impact = point["l_plus"] - point["l_minus"]
            sign = "+" if impact >= 0 else ""
            
            # Display with color based on impact
            if impact > 0:
                st.markdown(f"- {filename}: <span style='color:#047857;'>{sign}{impact:.2f}</span>", unsafe_allow_html=True)
            elif impact < 0:
                st.markdown(f"- {filename}: <span style='color:#dc2626;'>{impact:.2f}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"- {filename}: {impact:.2f}")
    
    with evidence_tab:
        # Display detailed evidence cards for each data point
        st.subheader("Evidence Analysis")
        
        # Create a selectbox to choose which evidence to view
        evidence_names = [f"{point.get('filename', 'Unknown File')} - {datetime.fromisoformat(str(point['timestamp'])).strftime('%Y-%m-%d %H:%M')}" 
                          for point in data_points]
        
        selected_evidence = st.selectbox("Select evidence to view:", evidence_names)
        selected_index = evidence_names.index(selected_evidence)
        point = data_points[selected_index]
        
        st.markdown("---")
        st.subheader(f"Analysis of {point.get('filename', 'Unknown File')}")
        
        # Main statistics in a card
        st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
        
        # Main statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Log Likelihood (H):**")
            st.write(f"{point.get('l_plus', 0):.4f}")
        with col2:
            st.write("**Log Likelihood (¬H):**")
            st.write(f"{point.get('l_minus', 0):.4f}")
        with col3:
            impact = point.get('l_plus', 0) - point.get('l_minus', 0)
            st.write("**Net Impact:**")
            if impact > 0:
                st.markdown(f"<span style='color:#047857;'>+{impact:.4f}</span>", unsafe_allow_html=True)
            elif impact < 0:
                st.markdown(f"<span style='color:#dc2626;'>{impact:.4f}</span>", unsafe_allow_html=True)
            else:
                st.write(f"{impact:.4f}")
        
        # Probability info
        st.write("**Prior Probability:**", f"{point.get('probability', 0) - (point.get('l_plus', 0) - point.get('l_minus', 0)):.2%}")
        st.write("**Posterior Probability:**", f"{point.get('probability', 0):.2%}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # What to show section
        show_options = []
        if point.get('confidence_assessment'):
            show_options.append("Confidence Assessment")
        if 'analysis_code' in point:
            show_options.append("Analysis Code")
        if 'analysis_rationale' in point:
            show_options.append("Analysis Rationale")
        
        if show_options:
            selected_view = st.radio("Show:", show_options)
            
            if selected_view == "Confidence Assessment" and point.get('confidence_assessment'):
                st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
                st.write("### Confidence Assessment")
                
                confidence = point['confidence_assessment'].get('confidence_score', 0)
                confidence_class = ""
                if confidence > 0.7:
                    confidence_class = "confidence-high"
                elif confidence > 0.4:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
                
                st.markdown(f"<span class='{confidence_class}'>{confidence:.0%} Confidence</span>", unsafe_allow_html=True)
                
                # Display strengths and limitations
                if 'key_strengths' in point['confidence_assessment'] and point['confidence_assessment']['key_strengths']:
                    st.write("**Key Strengths:**")
                    for strength in point['confidence_assessment']['key_strengths']:
                        st.markdown(f"- {strength}")
                
                if 'key_limitations' in point['confidence_assessment'] and point['confidence_assessment']['key_limitations']:
                    st.write("**Key Limitations:**")
                    for limitation in point['confidence_assessment']['key_limitations']:
                        st.markdown(f"- {limitation}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif selected_view == "Analysis Code" and 'analysis_code' in point:
                st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
                st.write("### Analysis Code")
                st.code(point['analysis_code'], language="python")
                st.markdown('</div>', unsafe_allow_html=True)
                
            elif selected_view == "Analysis Rationale" and 'analysis_rationale' in point:
                st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
                st.write("### Analysis Rationale")
                st.write(point['analysis_rationale'])
                st.markdown('</div>', unsafe_allow_html=True)
    
    with visual_tab:
        # Render HTML visualization
        st.subheader("Interactive Visualization")
        
        # Create a temporary directory to store the HTML file
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Prepare the document for rendering
                # Add necessary fields expected by the renderer
                render_doc = hypothesis_entry.copy()
                
                # Ensure we have a valid prior_log_odds value
                render_doc["prior_log_odds"] = 0.0  # Default to even odds if not present
                
                # Set data points with the right structure for the renderer
                if "node_metadata" in render_doc and "data_points" in render_doc["node_metadata"]:
                    # Create a properly structured data_points array for the renderer
                    prepared_data_points = []
                    
                    for point in render_doc["node_metadata"]["data_points"]:
                        # Ensure all required fields have valid values
                        prepared_point = {
                            "filename": point.get("filename", "Unknown File"),
                            "posterior": point.get("new_posterior", 0.0),
                            "l_plus": point.get("l_plus", 0.0),
                            "l_minus": point.get("l_minus", 0.0),
                            "timestamp": point.get("timestamp", datetime.now()),
                            "metadata": {
                                "filename": point.get("filename", "Unknown File")
                            },
                            "analysis_rationale": point.get("analysis_rationale", "No rationale provided")
                        }
                        
                        # Add confidence assessment if available
                        if "confidence_assessment" in point:
                            prepared_point["confidence_assessment"] = point["confidence_assessment"]
                        else:
                            # Add default confidence assessment
                            prepared_point["confidence_assessment"] = {
                                "confidence_score": 0.5,
                                "explanation": "No confidence assessment available",
                                "key_strengths": [],
                                "key_limitations": []
                            }
                        
                        prepared_data_points.append(prepared_point)
                    
                    render_doc["data_points"] = prepared_data_points
                else:
                    # Create empty data points if none exist
                    render_doc["data_points"] = []
                
                # Add probability field (current posterior as probability)
                render_doc["probability"] = probability
                
                # Add confidence interval
                render_doc["confidence_interval"] = [lower, upper]
                
                # Initialize the renderer and generate HTML
                renderer = MongoInFactRenderer()
                html_output = renderer.render_mongo_analysis(render_doc)
                
                # Save HTML to a temporary file
                html_temp_path = os.path.join(temp_dir, f"hypothesis_{hypothesis_id}.html")
                with open(html_temp_path, 'w', encoding='utf-8') as f:
                    f.write(html_output)
                
                # For download button
                st.session_state["html_output"] = html_output
                
                # Display HTML using iframe
                # Read the HTML file and convert to base64
                with open(html_temp_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                html_base64 = base64.b64encode(html_content.encode()).decode()
                
                # Display in iframe
                st.markdown(
                    f'<iframe src="data:text/html;base64,{html_base64}" class="iframe-container"></iframe>',
                    unsafe_allow_html=True
                )
                
                # Provide download option
                html_filename = f"hypothesis_{hypothesis_id}.html"
                st.download_button(
                    label="Download HTML Report",
                    data=html_output,
                    file_name=html_filename,
                    mime="text/html"
                )
                
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
                st.write("Detailed error information:")
                st.code(str(e))
                
                # Display raw data to help debug
                with st.expander("Debug Data"):
                    st.write("Hypothesis Entry:")
                    st.json(hypothesis_entry)
    
    # Generate summary report if requested
    if st.button("Generate Summary Report"):
        with st.spinner("Generating report..."):
            try:
                model = st.session_state.get("model", None)
                api_key = st.session_state.get("api_key", None)
                provider = st.session_state.get("provider", None)
                
                if not provider or not model or not api_key:
                    st.warning("LLM model information not found. Please ensure provider, model and API key are set.")
                else:
                    # Create a prompt to generate a summary
                    prompt = f"""
                    Generate a detailed summary report for the following hypothesis:
                    
                    Hypothesis: {hypothesis_text}
                    
                    Current probability: {probability:.2%}
                    95% confidence interval: ({lower:.2%}, {upper:.2%})
                    Interpretation: {interpretation}
                    
                    Evidence analyzed:
                    """
                    
                    # Add information about each piece of evidence
                    for point in data_points:
                        filename = point.get("filename", "Unknown File")
                        impact = point.get("l_plus", 0) - point.get("l_minus", 0)
                        prompt += f"\n- {filename}: Impact on log odds: {impact:.4f}"
                    
                    prompt += """
                    
                    Please include in your summary:
                    1. A clear interpretation of the current probability
                    2. An analysis of the strength of evidence
                    3. Key limitations or uncertainties in the analysis
                    4. Suggestions for what additional evidence would be valuable
                    5. A conclusion about the hypothesis
                    
                    Format the response as a professional report with sections.
                    """
                    
                    # Call the LLM
                    summary_report = call_llm(provider, api_key, model, prompt)
                    
                    # Display the generated report
                    st.subheader("Summary Report")
                    st.markdown(summary_report)
                    
                    # Add download button for the report
                    report_filename = f"hypothesis_{hypothesis_id}_report.md"
                    st.download_button(
                        label="Download Report",
                        data=summary_report,
                        file_name=report_filename,
                        mime="text/markdown"
                    )
            
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    # Navigation buttons
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Analysis"):
            return "back"
    
    with col2:
        if st.button("Start New Analysis", type="primary"):
            # Clean up session state for a new analysis
            keys_to_clear = [
                "hypothesis_id", "current_file_id", "current_filename", 
                "parsed_data", "generated_code", "current_code", 
                "validated_code", "l_plus", "l_minus", 
                "simple_analysis", "tech_analysis", "html_output"
            ]
            
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            return "restart"
    
    return None