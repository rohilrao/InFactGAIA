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
    st.header("Step 5: Interactive Code Review")
    
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
    
    # Get file ID if it exists in session state
    file_id = st.session_state.get("current_file_id", None)
    
    if not file_id:
        # Let user select an unprocessed file if not already selected
        unprocessed_files = list(db.fs.files.find({"metadata.hypothesis_id": hypothesis_id, "status": "unprocessed"}))
        
        if not unprocessed_files:
            st.warning("No unprocessed files found. Please upload files in Step 4.")
            
            if st.button("← Back to Step 4"):
                return "back"
            st.stop()
        
        st.subheader("Select a file to analyze")
        file_options = {file["filename"]: file["_id"] for file in unprocessed_files}
        selected_filename = st.selectbox("Choose file:", list(file_options.keys()))
        
        if st.button("Select File"):
            st.session_state["current_file_id"] = file_options[selected_filename]
            st.session_state["current_filename"] = selected_filename
            return "reload"
    
    else:
        # We have a file to analyze
        filename = st.session_state.get("current_filename", "Selected File")
        st.subheader(f"Analyzing: {filename}")
        
        # Check if we already have parsed data and node
        parsed_data = st.session_state.get("parsed_data", None)
        node = st.session_state.get("node", None)
        
        if not parsed_data or not node:
            # Fetch file content and create temporary file
            file_obj = fs.get(ensure_object_id(file_id))
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, filename)
            
            with open(temp_file_path, "wb") as f:
                f.write(file_obj.read())
                
            # Initialize node and process file to extract data
            with st.spinner("Initializing and parsing file data..."):
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
                    
                    # Process data interactively to get parsed data
                    parsed_data, metadata = node.process_data_interactively(temp_file_path)
                    
                    # Store in session state
                    st.session_state["parsed_data"] = parsed_data
                    st.session_state["metadata"] = metadata
                    st.session_state["node"] = node
                    st.success("File parsed successfully!")
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        st.warning(f"Failed to remove temporary file: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error parsing file: {str(e)}")
                    st.stop()
        
        # Display parsed data in collapsible section
        with st.expander("Parsed Data (Click to expand)"):
            st.json(parsed_data)
        
        # Check if we have generated code
        generated_code = st.session_state.get("generated_code", None)
        
        if not generated_code:
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
                        st.success("Code generated successfully!")
                        return "reload"
                    except Exception as e:
                        st.error(f"Error generating code: {str(e)}")
        
        else:
            # Display the code in an editable text area
            st.subheader("Review and Edit Analysis Code")
            edited_code = st.text_area("Analysis Code", value=generated_code, height=400)
            
            # Check if code has been modified
            if edited_code != generated_code:
                st.session_state["generated_code"] = edited_code
                st.info("Code has been modified. Please verify it before executing.")
            
            # Allow user to provide feedback and regenerate code
            feedback_col1, feedback_col2 = st.columns([3, 1])
            
            with feedback_col1:
                feedback = st.text_area("Feedback for code improvement (optional)", 
                                         placeholder="Provide feedback on what to improve...")
            
            with feedback_col2:
                if st.button("Regenerate Code"):
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
                                {edited_code}
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
                                try:
                                    if hasattr(node, "_get_message_text"):
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
                                        elif provider.lower() == "gpt":
                                            response = node.client.chat.completions.create(
                                                model=node.model,
                                                max_tokens=8192,
                                                temperature=0.1,
                                                messages=[{"role": "user", "content": feedback_prompt}]
                                            )
                                            response_text = response.choices[0].message.content
                                        elif provider.lower() == "deepseek":
                                            # Assuming DeepSeek has a similar API structure
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
                                    else:
                                        # Fallback to a provider-agnostic method if available
                                        response_text = node.generate_text(feedback_prompt)
                                except Exception as e:
                                    st.error(f"Error generating response: {str(e)}")
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
                                st.success("Code regenerated successfully!")
                                return "reload"
                                
                            except Exception as e:
                                st.error(f"Error regenerating code: {str(e)}")
                    else:
                        st.warning("Please provide feedback to guide code regeneration.")
            # Execute code button
            st.subheader("Validate Code")
            execute_col1, execute_col2 = st.columns([1, 1])
            
            with execute_col1:
                if st.button("Validate and Test Code"):
                    with st.spinner("Testing code execution..."):
                        try:
                            # Get the node
                            node = st.session_state.get("node")
                            if not node:
                                st.error("Session expired. Please start over.")
                                st.stop()
                            
                            # Execute code
                            l_plus, l_minus = node.execute_analysis_code(edited_code, parsed_data)
                            
                            st.session_state["l_plus"] = l_plus
                            st.session_state["l_minus"] = l_minus
                            st.session_state["validated_code"] = edited_code
                            
                            # Convert log odds to probability for display
                            p_h_given_d = 1 / (1 + math.exp(-l_plus + l_minus))
                            
                            st.success("Code executed successfully!")
                            st.write(f"**l_plus (log P(data | hypothesis)):** {l_plus:.4f}")
                            st.write(f"**l_minus (log P(data | not hypothesis)):** {l_minus:.4f}")
                            st.write(f"**Probability of hypothesis given this data:** {p_h_given_d:.2%}")
                            
                        except Exception as e:
                            st.error(f"Code execution failed: {str(e)}")
                            st.info("Please revise the code and try again.")
            
            # If code has been validated, allow proceeding to next step
            if "validated_code" in st.session_state:
                with execute_col2:
                    if st.button("Continue to Processing"):
                        # Store necessary information in session state
                        st.session_state["file_ready_for_processing"] = True
                        return "next"
    
    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Step 4"):
            # Clean up session state
            for key in ["current_file_id", "current_filename", "parsed_data", 
                       "generated_code", "node", "validated_code", "l_plus", "l_minus"]:
                if key in st.session_state:
                    del st.session_state[key]
                    
            return "back"
    
    return None  # No action taken