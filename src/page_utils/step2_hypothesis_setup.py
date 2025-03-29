import streamlit as st
import time
import uuid
import os
import json
import sys
import tempfile
from pathlib import Path
import bson

# Get project root from session state (set in the main app file)
def get_project_root():
    if "project_root" in st.session_state:
        return st.session_state["project_root"]
    else:
        # Fallback if not set
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Add the infact module to path
project_root = get_project_root()
infact_path = os.path.join(project_root, "infact")
if infact_path not in sys.path:
    sys.path.append(infact_path)

# Import InFactNode and providers
try:
    from infact import InFactNode
    from infact.providers import AnthropicProvider, OpenAIProvider
except ImportError:
    st.error("Could not import InFactNode modules. Please ensure the infact package is installed correctly.")
    

def display_combined_hypothesis_step(hypothesis_collection, call_llm):
    """
    Combined function for hypothesis setup, description, summary and chat in a single step.
    Now integrates with InFactNode for state management.
    
    Args:
        hypothesis_collection: MongoDB collection for hypotheses
        call_llm: Function to call LLM API (will be replaced with InFactNode providers)
        
    Returns:
        str: Navigation action - "back", "next", or None
    """
    # Initialize session state variables if they don't exist
    if "id_exists" not in st.session_state:
        st.session_state["id_exists"] = None
    if "sections_expanded" not in st.session_state:
        st.session_state["sections_expanded"] = {
            "description": True,
            "background": False,
            "chat": False
        }
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_id" not in st.session_state:
        st.session_state["chat_id"] = str(uuid.uuid4())
    if "user_question" not in st.session_state:
        st.session_state["user_question"] = ""
    if "infact_node" not in st.session_state:
        st.session_state["infact_node"] = None
    if "temp_node_file" not in st.session_state:
        st.session_state["temp_node_file"] = None
        
    # Ensure the sections_expanded dictionary has all required keys
    if "background" not in st.session_state["sections_expanded"]:
        st.session_state["sections_expanded"]["background"] = False
    if "chat" not in st.session_state["sections_expanded"]:
        st.session_state["sections_expanded"]["chat"] = False
    if "description" not in st.session_state["sections_expanded"]:
        st.session_state["sections_expanded"]["description"] = True

    # CSS for dark mode compatible chat interface
    st.markdown("""
    <style>
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

    # Helper function to create InFactNode provider based on user settings
    def create_llm_provider():
        provider_name = st.session_state["provider"]
        model = st.session_state["model"]
        api_key = st.session_state["api_key"]
        
        if provider_name == "Anthropic":
            return AnthropicProvider(api_key=api_key, model=model)
        elif provider_name == "GPT":
            return OpenAIProvider(api_key=api_key, model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    # Function to create a temporary file from a node state JSON
    def create_temp_node_file(node_state_json):
        # Clean up old temporary file if it exists
        if st.session_state["temp_node_file"] and os.path.exists(st.session_state["temp_node_file"]):
            try:
                os.remove(st.session_state["temp_node_file"])
            except Exception as e:
                print(f"Failed to remove old temporary file: {e}")
        
        # Create a new temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='node_state_')
        os.close(fd)
        
        # Write the node state JSON to the temporary file
        with open(temp_path, 'w') as f:
            if isinstance(node_state_json, dict):
                json.dump(node_state_json, f, indent=2)
            else:
                f.write(node_state_json)
        
        # Store the path in session state
        st.session_state["temp_node_file"] = temp_path
        
        return temp_path

    # Function to check if ID exists in database and load the node state if available
    def check_hypothesis_id():
        _id = st.session_state.get("hypothesis_id_input", "").strip()
        if not _id:
            st.session_state["id_exists"] = None
            st.session_state["infact_node"] = None
            return

        # Look up in DB
        existing = hypothesis_collection.find_one({"_id": _id})
        st.session_state["id_exists"] = True if existing else False
        
        if existing:
            # Check if there's node state data in MongoDB
            if "node_state" in existing and existing["node_state"]:
                try:
                    # Get provider info
                    provider_name = st.session_state["provider"]
                    api_key = st.session_state["api_key"]
                    
                    # Create a temporary file with the node state data from MongoDB
                    node_state_json = existing["node_state"]
                    temp_node_file = create_temp_node_file(node_state_json)
                    
                    # Load the existing node state from the temporary file
                    st.session_state["infact_node"] = InFactNode.load(
                        filename=temp_node_file,
                        provider_type=provider_name,
                        api_key=api_key,
                        model=st.session_state["model"]
                    )
                    st.info(f"Loaded existing InFactNode state for hypothesis {_id}")
                except Exception as e:
                    st.warning(f"Failed to load existing InFactNode state: {str(e)}")
                    st.session_state["infact_node"] = None
            else:
                st.session_state["infact_node"] = None

    # Function to handle chat submission
    def handle_chat_submit():
        user_input = st.session_state["user_question"].strip()
        if user_input:
            # Add user message to chat history
            st.session_state["chat_history"].append({
                "role": "user",
                "content": user_input
            })
            
            # Store the question to process after rerun
            st.session_state["pending_question"] = user_input
            
            # Clear the input field by updating session state
            st.session_state["user_question"] = ""

    # 1. HYPOTHESIS SETUP SECTION
    st.markdown("### :orange[Step 2: Hypothesis Setup]")
    # Add explanatory text above the text input
    st.markdown("""
    Please enter a unique **Hypothesis ID** to identify your hypothesis. This ID will be used to associate all results and files with your hypothesis.  
    You can retrieve existing IDs from the Hypotheses Explorer page.  

    For example, if you want to test a hypothesis like *'Is the earth flat?'*, you could use an ID like `hyp_flat_earth`.  
    If the ID already exists, the associated hypothesis will be loaded. Otherwise, a new hypothesis will be created.
    """)
    with st.container():
        st.text_input(
            "Hypothesis ID (please enter a unique short name as an identifier)",
            key="hypothesis_id_input",
            on_change=check_hypothesis_id
        )

    # Get hypothesis ID and check existence
    hypothesis_id = st.session_state.get("hypothesis_id_input", "").strip()
    loaded_text = None
    hypothesis_doc = None
    
    if hypothesis_id and st.session_state["id_exists"] is True:
        # The ID exists in DB => fetch its data
        hypothesis_doc = hypothesis_collection.find_one({"_id": hypothesis_id})
        if hypothesis_doc:
            loaded_text = hypothesis_doc["text"]
            st.session_state["hypothesis_text_input"] = loaded_text
            
            # Show InFactNode state info if it exists
            if st.session_state["infact_node"]:
                infact_node = st.session_state["infact_node"]
                current_prob = 1 / (1 + 2.71828 ** -infact_node.current_posterior)
                st.success(f"""
                Loaded existing hypothesis with ID '{hypothesis_id}'.
                Current probability: {current_prob:.2%}
                Number of data points: {len(infact_node.data_points)}
                """)
            else:
                st.success(f"Loaded existing hypothesis with ID '{hypothesis_id}'.")
        else:
            st.error("Inconsistent state: ID exists but not found in DB.")
            st.stop()
    
    # Display appropriate text area based on ID existence
    with st.container():
        if st.session_state["id_exists"] is True and loaded_text:
            st.text_area(
                "Hypothesis Text (read-only)",
                value=loaded_text,
                disabled=True,
                height=80
            )
        else:
            if hypothesis_id and st.session_state["id_exists"] is False:
                st.warning("No existing hypothesis found for this ID. You can create a new one below.")
            
            st.text_area(
                "Hypothesis Text",
                key="hypothesis_text_input",
                placeholder="Enter your hypothesis details here",
                height=80
            )
            
            # Create Hypothesis Button
            if st.session_state["id_exists"] is False and hypothesis_id:
                if st.button("Create Hypothesis", key="create_hypothesis_btn"):
                    new_text = st.session_state.get("hypothesis_text_input", "").strip()
                    if not new_text:
                        st.warning("⚠️ Please enter text before creating a new hypothesis.")
                        st.stop()
                    
                    # Create a new InFactNode for this hypothesis
                    try:
                        llm_provider = create_llm_provider()
                        
                        new_node = InFactNode(
                            hypothesis=new_text,
                            llm_provider=llm_provider,
                            prior_log_odds=0.0  # Start with neutral prior
                        )
                        
                        # Create a temporary file for saving
                        temp_file = tempfile.mktemp(suffix='.json')
                        
                        # Save the initial node state
                        new_node.save(temp_file)
                        
                        # Read the temporary file and store its content in MongoDB
                        with open(temp_file, 'r') as f:
                            node_state_json = json.load(f)
                        
                        # Remove the temporary file
                        os.remove(temp_file)
                        
                        # Store the node in session state
                        st.session_state["infact_node"] = new_node
                        
                        # Insert new doc with node state JSON
                        hypothesis_collection.insert_one({
                            "_id": hypothesis_id,
                            "original_text": new_text,
                            "text": new_text,
                            "short_description": "",  # New field for editable description
                            "auto_summary": None,
                            "node_state": node_state_json  # Store the node state JSON directly in MongoDB
                        })

                        # Mark as created and store in session
                        st.session_state["hypothesis_id"] = hypothesis_id
                        st.session_state["hypothesis_text"] = new_text

                        # Force check so button disappears on rerun
                        check_hypothesis_id()

                        st.success(f"✅ Created new hypothesis with ID '{hypothesis_id}'")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating InFactNode: {str(e)}")
                        st.stop()

    # Only continue if we have a valid hypothesis
    can_proceed = (st.session_state["id_exists"] is True and hypothesis_id) or \
                  st.session_state.get("hypothesis_id") == hypothesis_id
    
    if not can_proceed:
        # Show navigation buttons and stop
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back", key="back_btn_early"):
                return "back"
        return None

    # Get the active hypothesis ID
    active_id = st.session_state.get("hypothesis_id", hypothesis_id)
    
    # Ensure we have the hypothesis document
    if not hypothesis_doc:
        hypothesis_doc = hypothesis_collection.find_one({"_id": active_id})
    
    if not hypothesis_doc:
        st.error("Could not retrieve hypothesis data.")
        return None

    # Get or create InFactNode
    infact_node = st.session_state.get("infact_node")
    if not infact_node:
        try:
            # Create a new provider
            llm_provider = create_llm_provider()
            
            # Get the hypothesis text
            hypothesis_text = hypothesis_doc.get("text", "")
            
            # Create new node
            infact_node = InFactNode(
                hypothesis=hypothesis_text,
                llm_provider=llm_provider,
                prior_log_odds=0.0  # Start with neutral prior
            )
            
            # Create a temporary file for saving
            temp_file = tempfile.mktemp(suffix='.json')
            
            # Save initial state
            infact_node.save(temp_file)
            
            # Read the temporary file and store its content in MongoDB
            with open(temp_file, 'r') as f:
                node_state_json = json.load(f)
            
            # Remove the temporary file
            os.remove(temp_file)
            
            # Update the document with the node state JSON
            hypothesis_collection.update_one(
                {"_id": active_id},
                {"$set": {"node_state": node_state_json}}
            )
            
            # Store in session state
            st.session_state["infact_node"] = infact_node
        except Exception as e:
            st.error(f"Error creating or retrieving InFactNode: {str(e)}")

    st.divider()
    
    # 2. HYPOTHESIS REFINEMENT SECTION
    st.markdown("### :orange[Hypothesis:]")   
    
    # Process for Yes/No reformulation if needed
    need_reformulation = False
    original_text = hypothesis_doc.get("original_text", hypothesis_doc["text"])
    
    if "original_text" not in hypothesis_doc or hypothesis_doc["original_text"] == hypothesis_doc["text"]:
        need_reformulation = True
    
    if need_reformulation:
        with st.spinner("Reformulating hypothesis as a yes-no question..."):
            prompt_reformulate = (
                f"Given this hypothesis:\n\n'{original_text}'\n\n"
                "Rewrite/Reformulate it as a clear, concise Yes-No question. " 
                "The answer to the reformulated question should be either 'Yes' or 'No'. "
                "Keep your response brief — ONLY return the reformulated question, nothing else."
            )
            
            try:
                # Use the InFactNode provider instead of call_llm
                if st.session_state["infact_node"]:
                    llm_provider = st.session_state["infact_node"].llm_provider
                    yes_no_formulation = llm_provider.send_message(prompt_reformulate).strip()
                else:
                    # Fall back to call_llm if InFactNode isn't available
                    yes_no_formulation = call_llm(
                        provider=st.session_state["provider"],
                        model=st.session_state["model"],
                        api_key=st.session_state["api_key"],
                        prompt_text=prompt_reformulate
                    ).strip()
                
                # Validate the response
                if not yes_no_formulation.endswith('?'):
                    yes_no_formulation = yes_no_formulation.rstrip('.') + '?'
                
                # Update DB
                hypothesis_collection.update_one(
                    {"_id": active_id},
                    {"$set": {
                        "original_text": original_text,
                        "text": yes_no_formulation
                    }}
                )
                
                # Refresh the data
                hypothesis_doc["original_text"] = original_text
                hypothesis_doc["text"] = yes_no_formulation
                st.session_state["hypothesis_text"] = yes_no_formulation
                
                # Update the InFactNode hypothesis if available
                if st.session_state["infact_node"]:
                    st.session_state["infact_node"].hypothesis = yes_no_formulation
                    
                    # Save the node state to a temporary file
                    temp_file = tempfile.mktemp(suffix='.json')
                    st.session_state["infact_node"].save(temp_file)
                    
                    # Read the temporary file and update the MongoDB document
                    with open(temp_file, 'r') as f:
                        node_state_json = json.load(f)
                    
                    # Update MongoDB with new node state
                    hypothesis_collection.update_one(
                        {"_id": active_id},
                        {"$set": {"node_state": node_state_json}}
                    )
                    
                    # Remove the temporary file
                    os.remove(temp_file)
            except Exception as e:
                st.error(f"Error reformulating hypothesis: {str(e)}")
    
    # Display the refined hypothesis
    st.info(f"**Refined Question:** {hypothesis_doc['text']}")
    
    st.divider()
    
    # 3. EDITABLE SHORT DESCRIPTION SECTION
    st.markdown("### :orange[Hypothesis Short Description]")
    
    # Get existing description or generate a new one if needed
    description = hypothesis_doc.get("short_description", "")
    
    if not description:
        with st.spinner("Generating suggested description..."):
            prompt_description = (
                f"Given this yes/no hypothesis question:\n\n'{hypothesis_doc['text']}'\n\n"
                "Create a concise 3-4 sentence description that includes:\n"
                "1. A restatement of the hypothesis as a yes-no question\n"
                "2. A brief summary of the current state of knowledge\n"
                "3. What kind of data would be relevant for evaluating this hypothesis\n\n"
                "Make it clear and succinct, suitable as a hypothesis description that a researcher might write."
            )
            
            try:
                # Use the InFactNode provider instead of call_llm
                if st.session_state["infact_node"]:
                    llm_provider = st.session_state["infact_node"].llm_provider
                    description = llm_provider.send_message(prompt_description).strip()
                else:
                    # Fall back to call_llm if InFactNode isn't available
                    description = call_llm(
                        provider=st.session_state["provider"],
                        model=st.session_state["model"],
                        api_key=st.session_state["api_key"],
                        prompt_text=prompt_description
                    ).strip()
                
                # Update DB with suggested description
                hypothesis_collection.update_one(
                    {"_id": active_id},
                    {"$set": {"short_description": description}}
                )
                
                # Update local copy
                hypothesis_doc["short_description"] = description
            except Exception as e:
                st.error(f"Error generating description: {str(e)}")
                description = "Error generating description. Please try again later."
    
    # Display editable text area for description
    st.caption("You can edit the short description below:")
    new_description = st.text_area(
        "Edit Description",
        value=description,
        height=150,
        key=f"description_edit_{active_id}",
        label_visibility="collapsed"
    )
    
    # Save edited description
    if new_description != description:
        if st.button("Save Description", key="save_description_btn"):
            hypothesis_collection.update_one(
                {"_id": active_id},
                {"$set": {"short_description": new_description}}
            )
            st.success("✅ Description saved successfully!")
            # Update local copy
            hypothesis_doc["short_description"] = new_description
    
    st.divider()
    
    # 4. BACKGROUND SUMMARY SECTION
    st.markdown("### :orange[Detailed Background Summary]")
    
    # Get the background expanded state with a default of False if not set
    background_expanded = st.session_state["sections_expanded"].get("background", False)
    
    # Use the safely retrieved value
    show_summary = st.checkbox("Show detailed background", value=background_expanded)
    
    # Update the session state
    st.session_state["sections_expanded"]["background"] = show_summary
    
    if show_summary:
        # Generate or retrieve summary
        if "auto_summary" not in hypothesis_doc or hypothesis_doc["auto_summary"] is None:
            with st.spinner("Generating detailed background summary..."):
                prompt_summary = (
                    f"Given this yes/no hypothesis question:\n\n'{hypothesis_doc['text']}'\n\n"
                    "Create a comprehensive background summary with the following structure:\n\n"
                    "## Current State of Knowledge\n"
                    "Provide a detailed paragraph summarizing what is currently known about this topic. Include specific references to key studies and exact page numbers when applicable.\n\n"
                    "## Key Controversies\n"
                    "List 2-3 main points of debate in the field regarding this question, with specific citations.\n\n"
                    "## Relevant Evidence\n"
                    "### Evidence Supporting the Hypothesis\n"
                    "- [Study 1] Brief description, with citation including exact page numbers\n"
                    "- [Study 2] Brief description, with citation including exact page numbers\n\n"
                    "### Evidence Against the Hypothesis\n"
                    "- [Study 1] Brief description, with citation including exact page numbers\n"
                    "- [Study 2] Brief description, with citation including exact page numbers\n\n"
                    "Make your response well-structured and include detailed citations. Use minimal formatting and avoid overuse of emojis or decorative elements."
                )
                
                try:
                    # Use the InFactNode provider
                    if st.session_state["infact_node"]:
                        llm_provider = st.session_state["infact_node"].llm_provider
                        llm_response = llm_provider.send_message(prompt_summary)
                    else:
                        # Fall back to call_llm if InFactNode isn't available
                        llm_response = call_llm(
                            provider=st.session_state["provider"],
                            model=st.session_state["model"],
                            api_key=st.session_state["api_key"],
                            prompt_text=prompt_summary
                        )
                    
                    hypothesis_collection.update_one(
                        {"_id": active_id},
                        {"$set": {"auto_summary": llm_response}}
                    )
                    
                    # Refresh the data
                    hypothesis_doc["auto_summary"] = llm_response
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    st.warning("Could not generate summary. Please try again later.")
        
        # Display the summary
        if hypothesis_doc.get("auto_summary"):
            st.markdown(hypothesis_doc["auto_summary"])
        else:
            st.warning("Summary not available")
    
    st.divider()
    
    # 5. CHAT WITH BACKGROUND SECTION
    st.markdown("### :orange[Chat with Background Knowledge]")
    
    # Similar safe approach for chat expanded state
    chat_expanded = st.session_state["sections_expanded"].get("chat", False)
    show_chat = st.checkbox("Show chat interface", value=chat_expanded)
    st.session_state["sections_expanded"]["chat"] = show_chat
    
    if show_chat:
        # Process any pending question from previous run
        if "pending_question" in st.session_state and st.session_state["pending_question"]:
            with st.spinner("Generating response..."):
                user_question = st.session_state["pending_question"]
                
                # Ensure auto_summary exists before using it
                if "auto_summary" not in hypothesis_doc or hypothesis_doc["auto_summary"] is None:
                    # Generate summary if not available
                    with st.spinner("Generating background knowledge first..."):
                        prompt_summary = (
                            f"Given this yes/no hypothesis question:\n\n'{hypothesis_doc['text']}'\n\n"
                            "Create a comprehensive background summary with the following structure:\n\n"
                            "## Current State of Knowledge\n"
                            "Provide a detailed paragraph summarizing what is currently known about this topic.\n\n"
                            "## Key Controversies\n"
                            "List 2-3 main points of debate in the field regarding this question.\n\n"
                            "## Relevant Evidence\n"
                            "Summarize supporting and opposing evidence.\n\n"
                            "Keep it concise but informative."
                        )
                        
                        # Use the InFactNode provider
                        llm_provider = st.session_state["infact_node"].llm_provider
                        llm_response = llm_provider.send_message(prompt_summary)
                        
                        hypothesis_collection.update_one(
                            {"_id": active_id},
                            {"$set": {"auto_summary": llm_response}}
                        )
                        
                        # Refresh the data
                        hypothesis_doc["auto_summary"] = llm_response
                
                context = hypothesis_doc["text"] + "\n\n" + hypothesis_doc["auto_summary"]
                
                prompt_chat = (
                    f"You are an assistant helping a researcher understand the background of a hypothesis.\n\n"
                    f"The hypothesis is: {hypothesis_doc['text']}\n\n"
                    f"Background information:\n{hypothesis_doc['auto_summary']}\n\n"
                    f"The researcher asks: {user_question}\n\n"
                    f"Provide a helpful, specific answer based on the background information. "
                    f"If the answer isn't covered in the background information, say so clearly "
                    f"and suggest what kinds of information might be needed to address the question."
                )
                
                # Use the InFactNode provider
                llm_provider = st.session_state["infact_node"].llm_provider
                ai_response = llm_provider.send_message(prompt_chat)
                
                # Add AI response to chat history
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                # Clear the pending question
                st.session_state["pending_question"] = ""
        
        # Display the chat messages using custom HTML
        chat_container = st.container()
        with chat_container:
            for message in st.session_state["chat_history"]:
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
        
        # Chat input with on_change callback to avoid the session state issue
        st.text_input(
            "Ask a question about this hypothesis",
            key="user_question",
            on_change=handle_chat_submit
        )
        
        # Clear chat button
        if st.session_state["chat_history"] and st.button("Clear Chat", key="clear_chat_btn"):
            st.session_state["chat_history"] = []
            st.session_state["chat_id"] = str(uuid.uuid4())
            st.rerun()
    
    st.divider()
    
    # Add a new section to display InFactNode state information
    st.markdown("### :orange[Hypothesis Current State]")
    if st.session_state["infact_node"]:
        node = st.session_state["infact_node"]
        current_prob = 1 / (1 + 2.71828 ** -node.current_posterior)
        
        # Calculate confidence interval
        lower, upper = node._calculate_uncertainty()
        
        st.metric("Current Probability", f"{current_prob:.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lower Bound (95% CI)", f"{lower:.2%}")
        with col2:
            st.metric("Upper Bound (95% CI)", f"{upper:.2%}")
        
        st.write(f"**Number of data points evaluated:** {len(node.data_points)}")
        
        # Display evidence summary if there are data points
        if node.data_points:
            st.markdown("#### Evidence Summary")
            for i, dp in enumerate(node.data_points):
                with st.expander(f"Evidence {i+1}"):
                    st.write(f"**Log likelihood ratio:** {dp['l_plus'] - dp['l_minus']:.2f}")
                    
                    if 'confidence_assessment' in dp and dp['confidence_assessment']:
                        confidence = dp['confidence_assessment']
                        st.write(f"**Confidence score:** {confidence.get('confidence_score', 'N/A')}")
                        st.write(f"**Explanation:** {confidence.get('explanation', 'No explanation provided')}")
                        
                        if 'key_strengths' in confidence and confidence['key_strengths']:
                            st.write("**Key strengths:**")
                            for strength in confidence['key_strengths']:
                                st.write(f"- {strength}")
                        
                        if 'key_limitations' in confidence and confidence['key_limitations']:
                            st.write("**Key limitations:**")
                            for limitation in confidence['key_limitations']:
                                st.write(f"- {limitation}")
    
    st.divider()
    
    # 6. NAVIGATION BUTTONS
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("← Back", key="back_btn_final"):
            return "back"

    with col2:
        if st.button("Next →", key="next_btn_final"):
            # IMPORTANT: Ensure the hypothesis ID is properly stored in session state
            st.session_state["hypothesis_id"] = active_id
            
            # Debug - show what we're storing
            print(f"DEBUG - Storing hypothesis ID in session: {active_id}")
            
            # Also store the hypothesis text
            if hypothesis_doc:
                st.session_state["hypothesis_text"] = hypothesis_doc["text"]
                print(f"DEBUG - Storing hypothesis text in session: {hypothesis_doc['text'][:30]}...")
            
            # Ensure the InFactNode is saved before proceeding
            if st.session_state["infact_node"]:
                node_state_path = hypothesis_doc.get("node_state_path")
                if node_state_path:
                    st.session_state["infact_node"].save(node_state_path)
            
            return "next"