import streamlit as st
import os
import tempfile
from bson.objectid import ObjectId
import math
from autogen.code_utils import extract_code

def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except:
            return id_value
    return id_value

def display_code_review_step(db, fs, hypothesis_collection):
    """
    Handles Step 5: Interactive Code Review
    
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
    }
    .analysis-box {
        height: 150px;
        overflow-y: auto;
        padding: 10px;
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .results-container {
        background-color: #f0f7ff; 
        padding: 15px; 
        border-radius: 5px; 
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
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
            "$or": [
                {"status": "ready_for_analysis"},
                {"parsing_complete": True}
            ]
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
            if file_obj and "parsed_data" in file_obj:
                st.session_state["parsed_data"] = file_obj["parsed_data"]
            
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
            if file_obj and "parsed_data" in file_obj:
                parsed_data = file_obj["parsed_data"]
                st.session_state["parsed_data"] = parsed_data
            else:
                st.error("Could not find parsed data for this file. Please return to the file upload step.")
                if st.button("← Back to File Upload"):
                    return "back"
                st.stop()
        
        # Display parsed data in collapsible section
        with st.expander("Parsed Data", expanded=False):
            st.json(parsed_data)
        
        # Check if we have a node for analysis
        node = st.session_state.get("node", None)
        
        if not node:
            with st.spinner("Initializing analysis environment..."):
                provider = st.session_state.get("provider", "anthropic")
                model = st.session_state.get("model", "claude-3-5-sonnet-20241022")
                api_key = st.session_state.get("api_key", "")
                
                try:
                    # Import necessary node classes
                    from AnthropicInFactNode import AnthropicInFactNode
                    from GptInFactNode import GptInFactNode
                    from DeepSeekInFactNode import DeepSeekInFactNode
                    
                    # Create appropriate node type
                    if provider.lower() == "anthropic":
                        node = AnthropicInFactNode(
                            hypothesis=hypothesis_text,
                            api_key=api_key,
                            model=model
                        )
                    elif provider.lower() == "gpt":
                        node = GptInFactNode(
                            hypothesis=hypothesis_text,
                            api_key=api_key,
                            model=model
                        )
                    elif provider.lower() == "deepseek":
                        node = DeepSeekInFactNode(
                            hypothesis=hypothesis_text,
                            api_key=api_key,
                            model=model
                        )
                    else:
                        st.error(f"Unknown provider: {provider}")
                        st.stop()
                    
                    # Store in session state
                    st.session_state["node"] = node
                    
                except Exception as e:
                    st.error(f"Error initializing analysis environment: {str(e)}")
                    st.stop()
        
        # Check if we have generated code
        generated_code = st.session_state.get("generated_code", None)
        
        if not generated_code:
            st.markdown("### :orange[Generate Analysis Code]")
            st.info("We'll generate code to analyze your data in relation to the hypothesis.")
            
            if st.button("Generate Analysis Code"):
                with st.spinner("Generating analysis code..."):
                    try:
                        # Get the node from session state
                        node = st.session_state.get("node")
                        if not node:
                            st.error("Session expired. Please start over.")
                            st.stop()
                        
                        # Generate code using our interactive function
                        code = node.interactive_analyze_data(parsed_data)
                        st.session_state["generated_code"] = code
                        st.session_state["current_code"] = code  # Track current version
                        st.success("Code generated successfully!")
                        return "reload"
                    except Exception as e:
                        st.error(f"Error generating code: {str(e)}")
        
        else:
            # Display the code in an editable text area with syntax highlighting
            st.markdown("### :orange[Review and Edit Analysis Code]")
            
            # Get the current working code (may be edited from original)
            current_code = st.session_state.get("current_code", generated_code)
            
            # Show editing tips
            st.info("✏️ You can edit the code below. For complex edits, consider copying to your IDE, then paste back here.")
            
            # Display code editor with streamlit-ace
            try:
                # Try to use streamlit-ace if available
                import streamlit_ace
                
                edited_code = streamlit_ace.st_ace(
                    value=current_code,
                    language="python",
                    theme="github",
                    min_lines=20,
                    max_lines=40,
                    key="ace_editor"
                )
            except ImportError:
                # Fallback to regular text area with custom styling
                st.markdown('<div class="code-editor">', unsafe_allow_html=True)
                edited_code = st.text_area(
                    "Analysis Code", 
                    value=current_code, 
                    height=400,
                    key="code_editor"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Save button for code edits
            code_changed = edited_code != current_code
            save_col, tips_col = st.columns([1, 3])
            
            with save_col:
                if code_changed:
                    if st.button("Save Code Changes"):
                        st.session_state["current_code"] = edited_code
                        st.success("Code changes saved!")
                        return "reload"
            
            with tips_col:
                if code_changed:
                    st.warning("⚠️ You have unsaved changes to the code.")
            
            # Provide a code analysis
            st.markdown("### :orange[Code Analysis]")
            
            # Generate a brief analysis if not already present or if code changed
            code_analysis = st.session_state.get("code_analysis", None)
            
            if not code_analysis or code_changed:
                with st.spinner("Analyzing code..."):
                    try:
                        node = st.session_state.get("node")
                        if not node:
                            st.error("Session expired. Please start over.")
                            st.stop()
                        
                        analysis_prompt = f"""
                        Analyze this Python code in the context of the data and hypothesis.
                        
                        Hypothesis: {hypothesis_text}
                        
                        Code:
                        ```python
                        {current_code if not code_changed else edited_code}
                        ```
                        
                        Provide a VERY BRIEF analysis (maximum 150 words) with these sections:
                        1. Strengths - what the code does well
                        2. Limitations - what could be improved
                        3. Key assumptions made by the code
                        
                        Keep your response extremely concise and focused on the most important points.
                        """
                        
                        # Generate response using the appropriate provider
                        provider = st.session_state.get("provider", "anthropic")
                        
                        if provider.lower() == "anthropic":
                            message = node.client.messages.create(
                                model=node.model,
                                max_tokens=500,
                                temperature=0,
                                messages=[{
                                    "role": "user",
                                    "content": analysis_prompt
                                }]
                            )
                            analysis = node._get_message_text(message)
                        elif provider.lower() in ["gpt", "deepseek"]:
                            response = node.client.chat.completions.create(
                                model=node.model,
                                max_tokens=500,
                                temperature=0,
                                messages=[{"role": "user", "content": analysis_prompt}]
                            )
                            analysis = response.choices[0].message.content
                        else:
                            analysis = "Code analysis not available for this provider."
                        
                        # Store if not temporary
                        if not code_changed:
                            st.session_state["code_analysis"] = analysis
                        
                        code_analysis = analysis
                        
                    except Exception as e:
                        code_analysis = "Error generating code analysis."
                        st.error(f"Error analyzing code: {str(e)}")
            
            # Display the analysis in a small scrollable box
            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
            st.markdown(code_analysis)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Feedback and code regeneration
            st.markdown("### :orange[Improve the Code]")
            
            feedback = st.text_area(
                "Feedback for code improvement", 
                placeholder="Provide specific feedback on what to improve in the code...",
                height=100
            )
            
            if st.button("Regenerate Code with Feedback"):
                if feedback:
                    with st.spinner("Regenerating code based on feedback..."):
                        try:
                            # Get the node and provider
                            node = st.session_state.get("node")
                            provider = st.session_state.get("provider", "anthropic")
                            
                            if not node:
                                st.error("Session expired. Please start over.")
                                st.stop()
                            
                            # Generate new code with feedback
                            feedback_prompt = f"""
                            Here is the original code:
                            
                            ```python
                            {current_code if not code_changed else edited_code}
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
                            
                            # Generate response using the appropriate provider
                            response_text = ""
                            if provider.lower() == "anthropic":
                                message = node.client.messages.create(
                                    model=node.model,
                                    max_tokens=8192,
                                    temperature=0.1,
                                    messages=[{
                                        "role": "user",
                                        "content": feedback_prompt
                                    }]
                                )
                                response_text = node._get_message_text(message)
                            elif provider.lower() in ["gpt", "deepseek"]:
                                response = node.client.chat.completions.create(
                                    model=node.model,
                                    max_tokens=8192,
                                    temperature=0.1,
                                    messages=[{"role": "user", "content": feedback_prompt}]
                                )
                                response_text = response.choices[0].message.content
                            else:
                                st.error(f"Unsupported provider: {provider}")
                                st.stop()
                            
                            # Extract code using autogen
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
                            
                            st.session_state["generated_code"] = improved_code
                            st.session_state["current_code"] = improved_code
                            st.session_state.pop("code_analysis", None)  # Clear old analysis
                            st.success("Code regenerated successfully!")
                            return "reload"
                            
                        except Exception as e:
                            st.error(f"Error regenerating code: {str(e)}")
                else:
                    st.warning("Please provide feedback to guide code regeneration.")
            
            # Validate and test code
            st.markdown("### :orange[Validate and Test Code]")
            
            if st.button("Validate and Test Current Code"):
                with st.spinner("Testing code execution..."):
                    try:
                        # Get the node
                        node = st.session_state.get("node")
                        if not node:
                            st.error("Session expired. Please start over.")
                            st.stop()
                        
                        # Get the current code (either saved or the original)
                        code_to_test = current_code if not code_changed else edited_code
                        
                        # If code changed but not saved, warn user
                        if code_changed:
                            st.warning("⚠️ Testing unsaved code changes. Consider saving first.")
                        
                        # Execute code
                        l_plus, l_minus = node.execute_analysis_code(code_to_test, parsed_data)
                        
                        st.session_state["l_plus"] = l_plus
                        st.session_state["l_minus"] = l_minus
                        st.session_state["validated_code"] = code_to_test
                        
                        # Convert log odds to probability for display
                        p_h_given_d = 1 / (1 + math.exp(-l_plus + l_minus))
                        
                        st.success("Code executed successfully!")
                        
                        # Display results in a nice formatted box
                        st.markdown('<div class="results-container">', unsafe_allow_html=True)
                        st.markdown("#### Analysis Results")
                        st.write(f"**l_plus (log P(data | hypothesis)):** {l_plus:.4f}")
                        st.write(f"**l_minus (log P(data | not hypothesis)):** {l_minus:.4f}")
                        st.write(f"**Probability of hypothesis given this data:** {p_h_given_d:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Code execution failed: {str(e)}")
                        st.info("Please revise the code and try again.")
            
            # If code has been validated, allow proceeding to next step
            if "validated_code" in st.session_state:
                st.success("✅ Code successfully validated - ready to continue")
                if st.button("Continue to Processing"):
                    # Store necessary information in session state
                    st.session_state["file_ready_for_processing"] = True
                    return "next"
    
    # Navigation
    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to File Upload"):
            # Clean up session state
            for key in ["current_file_id", "current_filename", "parsed_data", 
                       "generated_code", "current_code", "node", "validated_code", 
                       "l_plus", "l_minus", "code_analysis"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            return "back"
    
    return None  # No action taken