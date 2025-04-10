import streamlit as st
import os
import tempfile
from bson.objectid import ObjectId
import math
from datetime import datetime
import base64
import uuid
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

def handle_chat_submit():
    """Handle chat submission."""
    user_input = st.session_state["result_question"].strip()
    if user_input:
        # Add user message to chat history
        st.session_state["result_chat_history"].append({
            "role": "user",
            "content": user_input
        })
        
        # Store the question to process after rerun
        st.session_state["result_pending_question"] = user_input
        
        # Clear the input field
        st.session_state["result_question"] = ""

def generate_analysis_takeaways(hypothesis_entry):
    """Generate key takeaways based on the hypothesis analysis."""
    takeaways = []
    
    # Get data points and metadata
    data_points = hypothesis_entry["node_metadata"].get("data_points", [])
    current_posterior = hypothesis_entry["node_metadata"].get("current_posterior", 0.0)
    probability = _to_probability(current_posterior)
    
    # Takeaway 1: Evidence quantity assessment
    num_data_points = len(data_points)
    if num_data_points < 3:
        takeaways.append("Limited evidence base: Analysis is based on only a few sources, which may affect reliability. Consider adding more evidence files.")
    elif num_data_points < 6:
        takeaways.append("Moderate evidence base: Analysis includes several sources but could benefit from additional evidence for more robust conclusions.")
    else:
        takeaways.append("Strong evidence base: Analysis includes multiple sources providing a more comprehensive assessment.")
    
    # Takeaway 2: Evidence quality and consistency assessment
    log_odds_impacts = [abs(point.get("l_plus", 0) - point.get("l_minus", 0)) for point in data_points]
    
    if log_odds_impacts:
        avg_impact = sum(log_odds_impacts) / len(log_odds_impacts)
        max_impact = max(log_odds_impacts)
        min_impact = min(log_odds_impacts)
        consistency = max_impact - min_impact
        
        if consistency > 8:
            takeaways.append("Inconsistent evidence strength: There are large variations in how strongly different sources support or contradict the hypothesis. Review the strongest contradicting evidence.")
        elif avg_impact < 2:
            takeaways.append("Weak evidence impact: The available evidence has relatively little impact on the hypothesis. Consider seeking stronger evidence.")
        else:
            takeaways.append("Consistent evidence: The evidence sources show reasonable consistency in their impact on the hypothesis.")
    
    # Takeaway 3: Confidence assessment
    lower, upper = _calculate_uncertainty(current_posterior, data_points)
    confidence_range = upper - lower
    
    if confidence_range > 0.4:
        takeaways.append("High uncertainty: The wide confidence interval suggests that more evidence is needed to reach a confident conclusion.")
    elif confidence_range < 0.1 and probability > 0.9:
        takeaways.append("Strong confidence in conclusion: The narrow confidence interval and high probability provide strong support for the hypothesis.")
    elif confidence_range < 0.1 and probability < 0.1:
        takeaways.append("Strong confidence in rejection: The narrow confidence interval and low probability provide strong evidence against the hypothesis.")
    else:
        takeaways.append("Moderate confidence: The confidence interval suggests a reasonable level of certainty in the conclusion, but more evidence could strengthen the finding.")
    
    # Store the full hypothesis entry in session state for use in chat
    st.session_state["hypothesis_entry"] = hypothesis_entry
    
    return takeaways

def process_chat_message(user_question, hypothesis_text, data_points, probability, lower, upper, interpretation):
    """Process a chat message about the results and generate a response."""
    try:
        provider_name = st.session_state["provider"]
        api_key = st.session_state["api_key"]
        model = st.session_state["model"]

        # Include the full hypothesis JSON data for more context
        hypothesis_json = st.session_state.get("hypothesis_entry", {})
        
        # Prepare evidence summary for the prompt
        evidence_summary = ""
        for point in data_points:
            filename = point.get("filename", "Unknown File")
            impact = point.get("l_plus", 0) - point.get("l_minus", 0)
            evidence_summary += f"\n- {filename}: Impact on log odds: {impact:.4f}"

        prompt_chat = (
            f"You are an assistant helping a researcher interpret the results of a Bayesian hypothesis analysis.\n\n"
            f"The hypothesis is: {hypothesis_text}\n\n"
            f"Current analysis results:\n"
            f"- Current probability: {probability:.2%}\n"
            f"- 95% confidence interval: ({lower:.2%}, {upper:.2%})\n"
            f"- Interpretation: {interpretation}\n\n"
            f"Evidence analyzed:{evidence_summary}\n\n"
            f"Full hypothesis data: {str(hypothesis_json)}\n\n"
            f"The researcher asks: {user_question}\n\n"
            f"Provide a helpful, specific answer focused on interpreting these results and what they mean for the hypothesis. "
            f"Consider the strength of evidence, limitations, and what additional evidence might be valuable. "
            f"Use your knowledge of Bayesian analysis to explain concepts if needed."
        )
        
        return call_llm(provider_name, api_key, model, prompt_chat)
    
    except Exception as e:
        return f"Error processing your question: {str(e)}"


def initialize_chat_state():
    """Initialize all required session state variables for chat."""
    if "result_chat_history" not in st.session_state:
        st.session_state["result_chat_history"] = []
    if "result_chat_id" not in st.session_state:
        st.session_state["result_chat_id"] = str(uuid.uuid4())
    if "result_question" not in st.session_state:
        st.session_state["result_question"] = ""

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
    # Initialize chat state variables
    initialize_chat_state()
    
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
    .chat-container {
        margin-bottom: 20px;
    }
    .chat-message {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: rgba(98, 156, 246, 0.2);
        border: 1px solid rgba(98, 156, 246, 0.4);
        margin-left: 20%;
        margin-right: 2%;
    }
    .assistant-message {
        background-color: rgba(131, 131, 131, 0.2);
        border: 1px solid rgba(131, 131, 131, 0.4);
        margin-right: 20%;
        margin-left: 2%;
    }
    .message-content {
        margin-top: 5px;
    }
    .message-sender {
        font-weight: bold;
        font-size: 0.85em;
        opacity: 0.8;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### :orange[Hypothesis Results]")
    
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
    
    # Display just two tabs: Visual Report and Chat
    visual_tab, chat_tab = st.tabs(["Visual Report", "Chat with Results"])
    
    with visual_tab:
        # Render HTML visualization
        st.subheader("Interactive Visualization")
        
        # Create a temporary directory to store the HTML file
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Just use the hypothesis_entry directly
                render_doc = hypothesis_entry.copy()
                
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

    try:
        # Get current timestamp for versioning
        current_timestamp = datetime.now()
        
        # Store the HTML directly in the hypothesis document
        hypothesis_collection.update_one(
            {"_id": hypothesis_id},
            {
                "$set": {
                    "node_metadata.latest_rendered_html": html_output,
                    "node_metadata.rendered_at": current_timestamp
                }
            }
        )
        
        st.success("Visualization saved to hypothesis document for future access.")
    except Exception as e:
        st.warning(f"Could not save visualization to the hypothesis document: {str(e)}")
    
    with chat_tab:
        st.subheader("Discuss Analysis Results")
        st.markdown("Ask questions about what these results mean for your hypothesis. The assistant will help interpret the data and suggest next steps.")
        
        # Process any pending question from previous run
        if "result_pending_question" in st.session_state and st.session_state["result_pending_question"]:
            with st.spinner("Generating response..."):
                user_question = st.session_state["result_pending_question"]
                
                # Generate response
                ai_response = process_chat_message(
                    user_question,
                    hypothesis_text,
                    data_points,
                    probability,
                    lower,
                    upper,
                    interpretation
                )
                
                # Add AI response to chat history
                st.session_state["result_chat_history"].append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                # Clear the pending question
                st.session_state["result_pending_question"] = ""
        
        # Display the chat messages using custom HTML
        chat_container = st.container()
        with chat_container:
            for message in st.session_state["result_chat_history"]:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-sender">You</div>
                        <div class="message-content">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="message-sender">Assistant</div>
                        <div class="message-content">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Suggest some example questions
        if not st.session_state["result_chat_history"]:
            st.markdown("#### Example questions you might ask:")
            st.markdown("- What do these results mean for my hypothesis?")
            st.markdown("- How strong is the evidence so far?")
            st.markdown("- What additional evidence would be most valuable?")
            st.markdown("- What are the main limitations of this analysis?")
            st.markdown("- How should I interpret the confidence interval?")
        
        # Chat input with on_change callback and a send button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_input(
                "Ask a question about the analysis results",
                key="result_question",
                on_change=handle_chat_submit
            )
        with col2:
            send_button = st.button("Send", key="send_result_chat_btn")
            if send_button and st.session_state["result_question"].strip():
                handle_chat_submit()
                st.rerun()
        
        # Clear chat button
        if st.session_state["result_chat_history"] and st.button("Clear Chat", key="clear_result_chat_btn"):
            st.session_state["result_chat_history"] = []
            st.session_state["result_chat_id"] = str(uuid.uuid4())
            st.rerun()
    

    # Add a new section for key takeaways
    st.markdown("---")
    st.subheader("Key Takeaways")
    
    # Analyze the hypothesis data to generate takeaways
    takeaways = generate_analysis_takeaways(hypothesis_entry)
    
    # Display the takeaways normally without styled container
    for point in takeaways:
        st.markdown(f"• {point}")


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
                "result_chat_history", "result_chat_id", "result_question",
                "result_pending_question", "html_output"
            ]
            
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            return "restart"
    
    return None