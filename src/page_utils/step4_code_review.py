import streamlit as st
import os
import tempfile
from bson.objectid import ObjectId
import math
from autogen.code_utils import extract_code
import json
from datetime import datetime
from code_analyzer import analyze_data
from code_analyzer import _execute_code_with_debug
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

def display_code_review_step(db, fs, hypothesis_collection):
    """
    Handles Step 4: Interactive Code Review
    
    Args:
        db: MongoDB database connection
        fs: GridFS instance
        hypothesis_collection: MongoDB collection for hypotheses
        
    Returns:
        str: Navigation action - "back", "next", or None
    """
    # Add syntax highlighting and editor styling
    st.markdown("""
    <style>
    .code-editor {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        line-height: 1.5;
        background-color: #1e1e1e;
        color: #d4d4d4;
        border-radius: 4px;
        padding: 8px;
    }
    .analysis-box {
        height: 150px;
        overflow-y: auto;
        padding: 10px;
        background-color: rgba(70, 70, 70, 0.2);
        border: 1px solid #333;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .results-container {
        background-color: rgba(0, 100, 0, 0.1);
        padding: 15px; 
        border-radius: 5px; 
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .chat-input {
        border-left: 3px solid #0078D7;
        background-color: rgba(0, 120, 215, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    .validation-section {
        background-color: rgba(255, 193, 7, 0.1);
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        border-left: 4px solid #FFC107;
    }
    .section-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #ccc;
        margin-bottom: 10px;
    }
    .nav-buttons {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .small-header {
        font-size: 0.9em;
        color: #aaa;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### :orange[Interactive Code Review]")
    
    # Get hypothesis information
    hypothesis_id = st.session_state.get("hypothesis_id", None)
    if not hypothesis_id:
        st.warning("No Hypothesis ID found in session. Please go back to Step 2.")
        if st.button("← Back to Hypothesis Setup"):
            return "back"
        st.stop()

    # Get the current hypothesis text and file info from DB
    hypothesis_entry = hypothesis_collection.find_one({"_id": hypothesis_id})
    if not hypothesis_entry:
        st.error("Could not find hypothesis data in database.")
        if st.button("← Back to Hypothesis Setup"):
            return "back"
        st.stop()

    hypothesis_text = hypothesis_entry["text"]
    st.write(f"**Hypothesis ID:** `{hypothesis_id}`")
    st.write(f"**Hypothesis:** {hypothesis_text}")
    st.divider()
    
    
    
    # Get file ID if it exists in session state
    file_id = st.session_state.get("current_file_id", None)
    
    if not file_id:
        # Let user select a processed (ready for analysis) file if not already selected
        st.markdown("### :orange[Select Data for Analysis]")
        
        # Get files that are ready for analysis (have parsed data)
        ready_files = list(db.fs.files.find({
                    "metadata.hypothesis_id": hypothesis_id, 
                    "metadata.status": "ready_for_analysis"  # Only this specific status
                    }))
        
        if not ready_files:
            st.warning("No files ready for analysis. Please upload and process files in the previous step.")
            
            if st.button("← Back to File Upload"):
                return "back"
            st.stop()
        
        file_options = {file["filename"]: file["_id"] for file in ready_files}
        selected_filename = st.selectbox("Choose file:", list(file_options.keys()))
        
        if st.button("Select File"):
            st.session_state["current_file_id"] = file_options[selected_filename]
            st.session_state["current_filename"] = selected_filename
            
            # Fetch the parsed data directly from the database
            file_obj = db.fs.files.find_one({"_id": file_options[selected_filename]})
            if file_obj and "metadata" in file_obj and "parsed_data" in file_obj["metadata"]:
                st.session_state["parsed_data"] = file_obj["metadata"]["parsed_data"]
            
            return "reload"
    
    else:
        # We have a file to analyze
        filename = st.session_state.get("current_filename", "Selected File")
        st.markdown(f"### :orange[Analyzing: {filename}]")
        
        # Get parsed data from session or directly from the database
        parsed_data = st.session_state.get("parsed_data", None)
        
        if not parsed_data:
            # Try to get parsed data from the database
            file_obj = db.fs.files.find_one({"_id": ensure_object_id(file_id)})
            if file_obj and "metadata" in file_obj and "parsed_data" in file_obj["metadata"]:
                st.session_state["parsed_data"] = file_obj["metadata"]["parsed_data"]
            else:
                st.error("Could not find parsed data for this file. Please return to the file upload step.")
                if st.button("← Back to File Upload"):
                    return "back"
                st.stop()
        
        # Display parsed data in collapsible section
        with st.expander("Parsed Data", expanded=False):
            st.json(parsed_data)
        
        # Check if we have generated code
        generated_code = st.session_state.get("generated_code", None)
        
        if not generated_code:
            st.markdown("### :orange[Generate Analysis Code]")
            st.info("We'll generate code to analyze your data in relation to the hypothesis.")
            
            if st.button("Generate Analysis Code"):
                with st.spinner("Generating analysis code..."):
                    try:
                        # Import data_analyzer directly and use it
                        
                        # Use analyze_data to generate code without executing it
                        # We only want the 'code' part from the tuple (l_plus, l_minus, code)
                        # Get model, api_key, and provider from session_state
                        model = st.session_state.get("model", None)
                        api_key = st.session_state.get("api_key", None)
                        provider = st.session_state.get("provider", None)
                            
                        credentials = (provider, model, api_key) 
                        
                        _, _, code = analyze_data(parsed_data, hypothesis_text, credentials)
                        
                        st.session_state["generated_code"] = code
                        st.session_state["current_code"] = code  # Track current version
                        
                        # Store the code in the database associated with the file
                        db.fs.files.update_one(
                            {"_id": ensure_object_id(file_id)},
                            {"$set": {"analysis_code": code}}
                        )
                        
                        st.success("Code generated successfully!")
                        return "reload"
                    except Exception as e:
                        st.error(f"Error generating code: {str(e)}")
                        print(f"Error generating analysis code: {str(e)}")
        
        else:
            # Main Code Display and Editing Section
            st.markdown("### :orange[Analysis Code]")
            
            # Track edit mode state
            edit_mode = st.session_state.get("edit_mode", False)
            current_code = st.session_state.get("current_code", generated_code)
            validated_code = st.session_state.get("validated_code", None)
            
            # Display the code with syntax highlighting by default
            if not edit_mode:
                code_to_display = validated_code if validated_code else current_code
                st.code(code_to_display, language="python")
                
                if st.button("Edit Code"):
                    st.session_state["edit_mode"] = True
                    return "reload"
            else:
                # We're in edit mode - show the editor
                st.info("For complex edits, consider copying to your IDE, then paste back here.")
                
                # Small header for code editor
                st.markdown('<p class="small-header">code editor</p>', unsafe_allow_html=True)
                
                # Display code editor with streamlit-ace if available
                try:
                    # Try to use streamlit-ace if available
                    import streamlit_ace
                    
                    edited_code = streamlit_ace.st_ace(
                        value=current_code,
                        language="python",
                        theme="monokai",  # Dark theme
                        min_lines=20,
                        max_lines=40,
                        key="ace_editor",
                        font_size=14,
                        keybinding="vscode",
                        show_gutter=True,
                        wrap=True,
                        auto_update=True
                    )
                except ImportError:
                    # Fallback to regular text area with custom styling
                    st.markdown('<div class="code-editor">', unsafe_allow_html=True)
                    edited_code = st.text_area(
                        "", 
                        value=current_code, 
                        height=400,
                        key="code_editor",
                        label_visibility="collapsed"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Save button for code edits
                code_changed = edited_code != current_code
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Save Edits" if code_changed else "Exit Editor"):
                        if code_changed:
                            st.session_state["current_code"] = edited_code
                            st.session_state.pop("validated_code", None)  # Code changed, need to revalidate
                            
                            # Update the code in the database
                            db.fs.files.update_one(
                                {"_id": ensure_object_id(file_id)},
                                {"$set": {"analysis_code": edited_code}}
                            )
                        st.session_state["edit_mode"] = False
                        return "reload"
                
                with col2:
                    if code_changed:
                        st.warning("You have unsaved changes to the code.")
            
            # Tabs for code improvement or analysis - always visible
            improve_tab, analyze_tab = st.tabs(["Improve Code", "Scrutinize Generated Code"])
            
            with improve_tab:
                # Apply styling with CSS class to the container
                st.markdown("""
                <style>
                [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:nth-child(1) {
                    border-left: 3px solid #0078D7;
                    background-color: rgba(0, 120, 215, 0.1);
                    padding: 10px;
                    border-radius: 5px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Add the feedback container
                feedback_container = st.container()
                with feedback_container:
                    feedback = st.text_area(
                        "Describe what you'd like to change in the code",
                        placeholder="Describe improvements or changes needed...",
                        height=80
                    )
                    if st.button("Apply Changes"):
                        if feedback:
                            with st.spinner("Regenerating code based on feedback..."):
                                try:
                                    # Get current code (edited or saved)
                                    code_to_improve = edited_code if (edit_mode and code_changed) else current_code
                                    
                                    # Generate new code with feedback using the node's llm_provider
                                    feedback_prompt = f"""
                                    Here is the original code:
                                    
                                    ```python
                                    {code_to_improve}
                                    ```
                                    
                                    User feedback:
                                    {feedback}
                                    
                                    Please improve the code based on this feedback. The code should still:
                                    1. Be a function named `calculate_log_likelihoods`
                                    2. Take a single dict parameter and return a tuple of (l_plus, l_minus)
                                    3. Calculate log likelihoods for the hypothesis: "{hypothesis_text}"
                                    4. Be ready to execute as-is
                                    
                                    Return only the improved Python code.
                                    """
                                    
                                    model = st.session_state.get("model", None)
                                    api_key = st.session_state.get("api_key", None)
                                    provider = st.session_state.get("provider", None)
                                    credentials = (provider, model, api_key) 
                                    
                                    response_text = call_llm(provider, api_key, model, feedback_prompt)
                                    
                                    # Extract code from the response
                                    extracted_code = extract_code(response_text)
                                    
                                    if not extracted_code:
                                        st.error("No code block found in AI response")
                                        st.stop()
                                    
                                    # Get the first Python code block
                                    improved_code = None
                                    for lang, code_block in extracted_code:
                                        if lang.lower() in ['python', 'py', '']:
                                            improved_code = code_block
                                            break
                                    
                                    if not improved_code:
                                        st.error("No Python code block found in AI response")
                                        st.stop()
                                    
                                    # Update session state
                                    st.session_state["generated_code"] = improved_code
                                    st.session_state["current_code"] = improved_code
                                    st.session_state.pop("validated_code", None)  # Need to revalidate
                                    st.session_state.pop("simple_analysis", None)  # Clear old analysis
                                    st.session_state.pop("tech_analysis", None)  # Clear old analysis
                                    st.session_state["edit_mode"] = False  # Exit edit mode
                                    
                                    # Update the database
                                    db.fs.files.update_one(
                                        {"_id": ensure_object_id(file_id)},
                                        {"$set": {"analysis_code": improved_code}}
                                    )
                                    
                                    st.success("Code regenerated successfully!")
                                    return "reload"
                                    
                                except Exception as e:
                                    st.error(f"Error regenerating code: {str(e)}")
                        
                                    print(f"Error regenerating code: {str(e)}")
                        else:
                            st.warning("Please provide feedback to guide code regeneration.")
            
            with analyze_tab:
                # Get the code to analyze (edited or current)
                code_for_analysis = edited_code if (edit_mode and code_changed) else current_code
                
                # Checkbox expanders for simple and technical analysis
                simple_analysis = st.checkbox("Simple Analysis", value=False)
                if simple_analysis:
                    # Check if we have a cached analysis
                    if "simple_analysis" in st.session_state:
                        st.markdown(st.session_state["simple_analysis"])
                    else:
                        with st.spinner("Generating simple analysis..."):
                            try:
                                simple_analysis_prompt = f"""
                                Provide a very short, simple explanation of this Python code in relation to the hypothesis:
                                
                                Hypothesis: {hypothesis_text}
                                
                                Code:
                                ```python
                                {code_for_analysis}
                                ```
                                
                                In 3-5 sentences, explain:
                                1. What this code is trying to do in simple terms
                                2. The main strengths of this code
                                3. Any limitations or weaknesses
                                
                                Use non-technical language that a layperson would understand.
                                """
                                model = st.session_state.get("model", None)
                                api_key = st.session_state.get("api_key", None)
                                provider = st.session_state.get("provider", None)
                                    
                                credentials = (provider, model, api_key) 
                                
                                
                                simple_analysis_text = call_llm(provider, api_key, model, simple_analysis_prompt)
                                
                                st.session_state["simple_analysis"] = simple_analysis_text
                                st.markdown(simple_analysis_text)
                                
                            except Exception as e:
                                st.error(f"Error generating analysis: {str(e)}")
                                
                                print(f"Error generating simple analysis: {str(e)}")
                
                tech_analysis = st.checkbox("Technical Analysis", value=False)
                if tech_analysis:
                    # Check if we have a cached analysis
                    if "tech_analysis" in st.session_state:
                        st.markdown(st.session_state["tech_analysis"])
                    else:
                        with st.spinner("Generating technical analysis..."):
                            try:
                                tech_analysis_prompt = f"""
                                Provide a detailed technical analysis of this Python code implementing Bayesian analysis:
                                
                                ```python
                                {code_for_analysis}
                                ```
                                
                                Focus on:
                                1. Correct implementation of Bayesian log-likelihood calculation (l_plus, l_minus)
                                2. Statistical validity of assumptions
                                3. Numerical stability concerns
                                4. Edge case handling
                                5. Efficiency of implementation
                                
                                For the hypothesis: "{hypothesis_text}"
                                
                                Keep your response developer-focused, identifying specific technical issues.
                                """
                                
                                
                                tech_analysis_text = call_llm(provider, api_key, model, tech_analysis_prompt)
                                
                                st.session_state["tech_analysis"] = tech_analysis_text
                                st.markdown(tech_analysis_text)
                                
                            except Exception as e:
                                st.error(f"Error generating technical analysis: {str(e)}")
                               
                                print(f"Error generating technical analysis: {str(e)}")
            
            # Validation section - always visible
            st.markdown("### :orange[Validate and Test Code]")
            
            # Show validation status
            if "validated_code" in st.session_state:
                st.success("Code validated successfully")
            else:
                st.info("Code needs to be validated before proceeding")
            
            # Validation button and results
            if st.button("Test Code"):
                with st.spinner("Testing code execution..."):
                    try:
                        # Get the code to test
                        if edit_mode and code_changed:
                            st.warning("Testing unsaved code changes.")
                            code_to_test = edited_code
                        else:
                            code_to_test = current_code

                        model = st.session_state.get("model", None)
                        api_key = st.session_state.get("api_key", None)
                        provider = st.session_state.get("provider", None)
                            
                        credentials = (provider, model, api_key) 
                        
                        l_plus, l_minus = _execute_code_with_debug(code_to_test, parsed_data, credentials)
                        
                        st.session_state["l_plus"] = l_plus
                        st.session_state["l_minus"] = l_minus
                        st.session_state["validated_code"] = code_to_test
                        
                        # Retrieve current posterior from the hypothesis document in the database
                        hypothesis_doc = hypothesis_collection.find_one({"_id": hypothesis_id})
                        if hypothesis_doc and "node_metadata" in hypothesis_doc and "current_posterior" in hypothesis_doc["node_metadata"]:
                            current_posterior = hypothesis_doc["node_metadata"]["current_posterior"]
                        else:
                            # Default to 0 if current_posterior is not found (prior is even odds)
                            current_posterior = 0.0
                            st.warning("Could not find current posterior value. Using default of even odds (0.0).")
                        
                        # Calculate the new posterior by adding log-likelihood ratio to current posterior
                        new_posterior = current_posterior + l_plus - l_minus

                        # Get existing data points or initialize empty list
                        data_points = []
                        if hypothesis_doc and "node_metadata" in hypothesis_doc and "data_points" in hypothesis_doc["node_metadata"]:
                            data_points = hypothesis_doc["node_metadata"]["data_points"]


                        # Add the current analysis as a new data point
                        new_data_point = {
                            "file_id": file_id,
                            "filename": st.session_state.get("current_filename", "Unknown file"),
                            "l_plus": l_plus,
                            "l_minus": l_minus,
                            "timestamp": datetime.now()
                        }
                        data_points.append(new_data_point)
                        
                        # Calculate probability and uncertainty
                        probability = _to_probability(new_posterior)
                        lower, upper = _calculate_uncertainty(new_posterior, data_points)

                        st.success("Code executed successfully!")

                        # Show validated code in an expander
                        with st.expander("Finalized Analysis Code", expanded=False):
                            st.code(code_to_test, language="python")

                        # Display results
                        st.markdown("#### Analysis Results")
                        st.write(f"**l_plus (log P(data | hypothesis)):** {l_plus:.4f}")
                        st.write(f"**l_minus (log P(data | not hypothesis)):** {l_minus:.4f}")
                        st.write(f"**Current posterior log odds:** {current_posterior:.4f}")
                        st.write(f"**New posterior log odds:** {new_posterior:.4f}")
                        st.write(f"**Probability of hypothesis given this data:** {probability:.2%}")
                        st.write(f"**Confidence interval (95%):** ({lower:.2%}, {upper:.2%})")

                        # Store the new posterior in the session state for later use
                        st.session_state["new_posterior"] = new_posterior
                        
                        # Update the hypothesis document with the new posterior and data point
                        hypothesis_collection.update_one(
                            {"_id": hypothesis_id},
                            {"$set": {
                                "node_metadata.current_posterior": new_posterior,
                                "node_metadata.data_points": data_points
                            }}
                        )

                        # Update the database with validation results
                        db.fs.files.update_one(
                            {"_id": ensure_object_id(file_id)},
                            {"$set": {
                                "analysis_code": code_to_test,
                                "analysis_results": {
                                    "l_plus": l_plus,
                                    "l_minus": l_minus,
                                    "current_posterior": current_posterior,
                                    "new_posterior": new_posterior,
                                    "probability": probability,
                                    "confidence_lower": lower,
                                    "confidence_upper": upper
                                }
                            }}
                        )
                        
                        # Exit edit mode after validation if we're in it
                        if edit_mode:
                            st.session_state["edit_mode"] = False
                            return "reload"
                        
                    except Exception as e:
                        st.error(f"Code execution failed: {str(e)}")
                        st.info("Please revise the code and try again.")
                        print(f"Code execution failed: {str(e)}")
            # Navigation buttons
            st.divider()
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back"):
                    # Clean up session state for this step
                    for key in ["current_file_id", "current_filename", "parsed_data", 
                              "generated_code", "current_code", "validated_code", 
                              "l_plus", "l_minus", "edit_mode",
                              "simple_analysis", "tech_analysis"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    return "back"
            
            with col2:
                if "validated_code" in st.session_state:
                    if st.button("Continue →", type="primary"):
                        # Store necessary information in session state
                        st.session_state["file_ready_for_processing"] = True
                        return "next"
    
    return None  # No action taken