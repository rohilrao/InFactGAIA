import streamlit as st
import os
import json
import uuid
import time
from uuid import uuid4
from datetime import datetime
from infact_utils import call_llm

def check_and_load_hypothesis(hypothesis_collection):
    """Check if hypothesis ID exists and load its data if present."""
    _id = st.text_input("Enter Hypothesis ID")
    if not _id:
        return

    doc = hypothesis_collection.find_one({"_id": _id})
    if doc:
        st.session_state["id_exists"] = True
        st.success(f"Loaded existing hypothesis ID: {_id}")
    else:
        st.session_state["id_exists"] = False
        st.info(f"No entry found for hypothesis ID '{_id}'. A new hypothesis will be created later.")

# ========== FUNCTIONAL COMPONENTS ==========

def reformulate_hypothesis_as_yes_no(original_text, active_id, hypothesis_collection):
    """Reformulate a hypothesis as a yes/no question."""
    try:
        # Use the API directly without InFactNode
        provider_name = st.session_state["provider"]
        api_key = st.session_state["api_key"]
        model = st.session_state["model"]
        
        prompt_reformulate = (
            f"Given this hypothesis:\n\n'{original_text}'\n\n"
            "Rewrite/Reformulate it as a clear, concise Yes-No question. " 
            "The answer to the reformulated question should be either 'Yes' or 'No'. "
            "Keep your response brief — ONLY return the reformulated question, nothing else."
        )
        
        yes_no_formulation = call_llm(provider_name, api_key, model, prompt_reformulate)
        
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
        
        return yes_no_formulation
    except Exception as e:
        st.error(f"Error reformulating hypothesis: {str(e)}")
        return original_text

def generate_hypothesis_description(hypothesis_text, active_id, hypothesis_collection, call_llm=None):
    """Generate a short description for the hypothesis."""
    try:
        # Use the API directly
        provider_name = st.session_state["provider"]
        api_key = st.session_state["api_key"]
        model = st.session_state["model"]
        
        prompt_description = (
            f"Given this yes/no hypothesis question:\n\n'{hypothesis_text}'\n\n"
            "Create a concise 3-4 sentence description that includes:\n"
            "1. A restatement of the hypothesis as a yes-no question\n"
            "2. A brief summary of the current state of knowledge\n"
            "3. What kind of data would be relevant for evaluating this hypothesis\n\n"
            "Make it clear and succinct, suitable as a hypothesis description that a researcher might write."
        )
        
       
        description = call_llm(provider_name, api_key, model, prompt_description)   .strip()
        
        # Update DB
        hypothesis_collection.update_one(
            {"_id": active_id},
            {"$set": {"short_description": description}}
        )
        
        return description
            
    except Exception as e:
        st.error(f"Error generating description: {str(e)}")
        return ""

def generate_background_summary(hypothesis_text, active_id, hypothesis_collection, call_llm=None):
    """Generate a detailed background summary for the hypothesis."""
    try:
        provider_name = st.session_state["provider"]
        api_key = st.session_state["api_key"]
        model = st.session_state["model"]

        prompt_summary = (
            f"Given this yes/no hypothesis question:\n\n'{hypothesis_text}'\n\n"
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
        
        
        summary = call_llm(provider_name, api_key, model, prompt_summary)
        
        # Update DB
        hypothesis_collection.update_one(
            {"_id": active_id},
            {"$set": {"auto_summary": summary}}
        )
        
        return summary
            
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return ""

def process_chat_message(user_question, hypothesis_text, background_summary, call_llm=None):
    """Process a chat message and generate a response."""
    try:
        provider_name = st.session_state["provider"]
        api_key = st.session_state["api_key"]
        model = st.session_state["model"]

        prompt_chat = (
            f"You are an assistant helping a researcher understand the background of a hypothesis.\n\n"
            f"The hypothesis is: {hypothesis_text}\n\n"
            f"Background information:\n{background_summary}\n\n"
            f"The researcher asks: {user_question}\n\n"
            f"Provide a helpful, specific answer based on the background information. "
            f"If the answer isn't covered in the background information, say so clearly "
            f"and suggest what kinds of information might be needed to address the question."
        )
        
        return call_llm(provider_name, api_key, model, prompt_chat)
    
    except Exception as e:
        return f"Error processing your question: {str(e)}"

def handle_chat_submit():
    """Handle chat submission."""
    user_input = st.session_state["user_question"].strip()
    if user_input:
        # Add user message to chat history
        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_input
        })
        
        # Store the question to process after rerun
        st.session_state["pending_question"] = user_input
        
        # Clear the input field
        st.session_state["user_question"] = ""

def check_hypothesis_id(hypothesis_collection):
    """Check if a hypothesis ID exists in the database."""
    hypothesis_id = st.session_state.get("hypothesis_id_input", "").strip()
    if hypothesis_id:
        doc = hypothesis_collection.find_one({"_id": hypothesis_id})
        st.session_state["id_exists"] = bool(doc)
    else:
        st.session_state["id_exists"] = None

def create_hypothesis(hypothesis_id, hypothesis_text, hypothesis_collection):
    """Create a new hypothesis in the database."""
    try:
        # Create a new hypothesis document
        hypothesis_doc = {
            "_id": hypothesis_id,
            "text": hypothesis_text,
            "created_at": datetime.now().isoformat(),
            "original_text": hypothesis_text
        }
        
        # Insert into the database
        hypothesis_collection.insert_one(hypothesis_doc)
        
        return True
    except Exception as e:
        st.error(f"Error creating hypothesis: {str(e)}")
        return False

# ========== UI COMPONENTS ==========

def initialize_session_state():
    """Initialize all required session state variables."""
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
        
    # Ensure the sections_expanded dictionary has all required keys
    if "background" not in st.session_state["sections_expanded"]:
        st.session_state["sections_expanded"]["background"] = False
    if "chat" not in st.session_state["sections_expanded"]:
        st.session_state["sections_expanded"]["chat"] = False
    if "description" not in st.session_state["sections_expanded"]:
        st.session_state["sections_expanded"]["description"] = True

def setup_ui_styles():
    """Set up CSS styles for the UI."""
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

def render_hypothesis_setup(hypothesis_collection):
    """Render the hypothesis setup section."""
    st.markdown("### :orange[Step 2: Hypothesis Setup]")
    
    # Add explanatory text
    st.markdown("""
    Please enter a unique **Hypothesis ID** to identify your hypothesis. This ID will be used to associate all results and files with your hypothesis.  
    You can retrieve existing IDs from the Hypotheses Explorer page.  

    For example, if you want to test a hypothesis like *'Is the earth flat?'*, you could use an ID like `hyp_flat_earth`.  
    If the ID already exists, the associated hypothesis will be loaded. Otherwise, a new hypothesis will be created.
    """)
    
    # Hypothesis ID input
    st.text_input(
        "Hypothesis ID (please enter a unique short name as an identifier)",
        key="hypothesis_id_input",
        on_change=lambda: check_hypothesis_id(hypothesis_collection)
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
            st.success(f"Hypothesis loaded with ID '{hypothesis_id}'.")
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
                    
                    success = create_hypothesis(hypothesis_id, new_text, hypothesis_collection)
                    if success:
                        st.success(f"✅ Created new hypothesis with ID '{hypothesis_id}'")
                        # Force check so button disappears on rerun
                        check_hypothesis_id(hypothesis_collection)
                        time.sleep(1)
                        st.rerun()
    
    return hypothesis_id, hypothesis_doc

def render_hypothesis_refinement(hypothesis_id, hypothesis_doc, hypothesis_collection, call_llm=None):
    """Render the hypothesis refinement section."""
    if not hypothesis_id or not hypothesis_doc:
        return None
    
    st.divider()
    st.markdown("### :orange[Hypothesis:]")
    
    # Process for Yes/No reformulation if needed
    need_reformulation = False
    original_text = hypothesis_doc.get("original_text", hypothesis_doc["text"])
    
    if "original_text" not in hypothesis_doc or hypothesis_doc["original_text"] == hypothesis_doc["text"]:
        need_reformulation = True
    
    if need_reformulation:
        with st.spinner("Reformulating hypothesis as a yes-no question..."):
            yes_no_formulation = reformulate_hypothesis_as_yes_no(
                original_text, 
                hypothesis_id, 
                hypothesis_collection
            )
            
            # Refresh the data
            hypothesis_doc["original_text"] = original_text
            hypothesis_doc["text"] = yes_no_formulation
            st.session_state["hypothesis_text"] = yes_no_formulation
    
    # Display the refined hypothesis
    st.info(f"**Refined Question:** {hypothesis_doc['text']}")
    
    return hypothesis_doc

def render_description_section(hypothesis_id, hypothesis_doc, hypothesis_collection, call_llm=None):
    """Render the description section."""
    if not hypothesis_id or not hypothesis_doc:
        return hypothesis_doc
    
    st.divider()
    st.markdown("### :orange[Hypothesis Short Description]")
    
    # Get existing description or generate a new one if needed
    description = hypothesis_doc.get("short_description", "")
    
    if not description:
        with st.spinner("Generating suggested description..."):
            description = generate_hypothesis_description(
                hypothesis_doc['text'],
                hypothesis_id,
                hypothesis_collection,
                call_llm
            )
            
            # Update local copy
            hypothesis_doc["short_description"] = description
    
    # Display editable text area for description
    st.caption("You can edit the short description below:")
    new_description = st.text_area(
        "Edit Description",
        value=description,
        height=150,
        key=f"description_edit_{hypothesis_id}",
        label_visibility="collapsed"
    )
    
    # Save edited description
    if new_description != description:
        if st.button("Save Description", key="save_description_btn"):
            hypothesis_collection.update_one(
                {"_id": hypothesis_id},
                {"$set": {"short_description": new_description}}
            )
            st.success("✅ Description saved successfully!")
            # Update local copy
            hypothesis_doc["short_description"] = new_description
    
    return hypothesis_doc

def render_background_section(hypothesis_id, hypothesis_doc, hypothesis_collection, call_llm=None):
    """Render the background summary section."""
    if not hypothesis_id or not hypothesis_doc:
        return hypothesis_doc
    
    st.divider()
    st.markdown("### :orange[Detailed Background Summary]")
    
    # Get the background expanded state
    background_expanded = st.session_state["sections_expanded"].get("background", False)
    show_summary = st.checkbox("Show detailed background", value=background_expanded)
    st.session_state["sections_expanded"]["background"] = show_summary
    
    if show_summary:
        # Generate or retrieve summary
        if "auto_summary" not in hypothesis_doc or hypothesis_doc["auto_summary"] is None:
            with st.spinner("Generating detailed background summary..."):
                summary = generate_background_summary(
                    hypothesis_doc['text'],
                    hypothesis_id,
                    hypothesis_collection,
                    call_llm
                )
                
                # Refresh the data
                hypothesis_doc["auto_summary"] = summary
        
        # Display the summary
        if hypothesis_doc.get("auto_summary"):
            st.markdown(hypothesis_doc["auto_summary"])
        else:
            st.warning("Summary not available")
    
    return hypothesis_doc

def render_chat_section(hypothesis_id, hypothesis_doc, hypothesis_collection, call_llm=None):
    """Render the chat section."""
    if not hypothesis_id or not hypothesis_doc:
        return
    
    st.divider()
    st.markdown("### :orange[Chat with Background Knowledge]")
    
    # Get chat expanded state
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
                        summary = generate_background_summary(
                            hypothesis_doc['text'],
                            hypothesis_id,
                            hypothesis_collection,
                            call_llm
                        )
                        
                        # Refresh the data
                        hypothesis_doc["auto_summary"] = summary
                
                # Generate response
                ai_response = process_chat_message(
                    user_question,
                    hypothesis_doc["text"],
                    hypothesis_doc["auto_summary"],
                    call_llm
                )
                
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
        
        # Chat input with on_change callback
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

def render_current_state(hypothesis_id):
    """Render the current state of the hypothesis."""
    st.divider()
    st.markdown("### :orange[Hypothesis Current State]")
    
    # Display basic information about the hypothesis
    st.write(f"**Hypothesis ID:** {hypothesis_id}")
    st.write("No evidence has been evaluated yet. Please proceed to the next step to start evaluating evidence.")

def render_navigation_buttons(active_id, hypothesis_doc):
    """Render navigation buttons."""
    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("← Back", key="back_btn_final"):
            return "back"

    with col2:
        if st.button("Next →", key="next_btn_final"):
            # Store hypothesis ID in session state
            st.session_state["hypothesis_id"] = active_id
            
            # Store the hypothesis text
            if hypothesis_doc:
                st.session_state["hypothesis_text"] = hypothesis_doc["text"]
            
            return "next"
    
    return None