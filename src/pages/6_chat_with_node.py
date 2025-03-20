import streamlit as st
from pymongo import MongoClient
import gridfs
import json
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi
import openai  # Or Anthropic API, DeepSeek API, etc.

# 🔹 **Define Chat Function FIRST to Avoid NameError**
def chat_with_node(api_key, provider, model, node_state_content, user_input):
    """
    Queries the LLM (GPT, Claude, or DeepSeek) with the node state and user's question.
    Restricts responses to the node state contents only.
    """
    system_prompt = f"""
    You are an AI assistant that interacts only with the provided node state.
    You must answer questions **only based on this JSON data** and refuse any off-topic conversations.
    Node State:
    {json.dumps(node_state_content, indent=2)}
    """

    if provider == "GPT":
        from openai import OpenAI  # ✅ Ensure using OpenAI's latest SDK

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content

    return "🛠️ AI model integration for this provider is under development."

# 🔐 MongoDB Connection
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["infact_db_v3"]
fs = gridfs.GridFS(db)

# 📌 UI - Chat with Node
st.title("💬 Chat with Node")

# ✅ Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "sections_expanded" not in st.session_state:
    st.session_state["sections_expanded"] = {"hypothesis": True, "file": True, "model": True, "chat": False}

# Step 1: User enters Hypothesis ID (Collapsible)
with st.expander("📌 Enter Hypothesis ID", expanded=st.session_state["sections_expanded"]["hypothesis"]):
    hypothesis_id = st.text_input("Enter Hypothesis ID:")
    if st.button("🔍 Load Processed Files"):
        if not hypothesis_id.strip():
            st.warning("⚠️ Please enter a valid Hypothesis ID.")
        else:
            st.session_state["loaded_hypothesis_id"] = hypothesis_id  # ✅ Store ID
            st.session_state.pop("selected_file_id", None)  # ✅ Reset file selection
            st.session_state["chat_history"] = []  # ✅ Reset chat history

# Get stored Hypothesis ID after button click
loaded_hypothesis_id = st.session_state.get("loaded_hypothesis_id", None)

if loaded_hypothesis_id:
    processed_files = list(fs.find({"hypothesis_id": loaded_hypothesis_id, "status": "processed"}))
    file_options = {str(f._id): f.filename for f in processed_files}

    if processed_files:
        # Step 2: Select a Processed File (Collapsible)
        with st.expander("📂 Select a Processed File", expanded=st.session_state["sections_expanded"]["file"]):
            selected_file_id = st.selectbox("Select a file:", list(file_options.keys()), format_func=lambda x: file_options[x])
            
            if st.button("📥 Load Node State"):
                if not selected_file_id:
                    st.warning("⚠️ Please select a processed file.")
                else:
                    st.session_state["selected_file_id"] = selected_file_id  # Store file ID in session

        # Step 3: Load Node State
        selected_file_id = st.session_state.get("selected_file_id", None)

        if selected_file_id:
            file_meta = next((f for f in processed_files if str(f._id) == selected_file_id), None)  # ✅ Prevents StopIteration

            if file_meta is None:
                st.warning("⚠️ The previously selected file is no longer available. Please select a new one.")
                st.session_state.pop("selected_file_id", None)  # ✅ Reset selection
            else:
                node_state_id = getattr(file_meta, "node_state_file_id", None)

            # ✅ Check if `node_state_id` exists in GridFS
            if node_state_id and fs.find_one({"_id": ObjectId(node_state_id)}):
                node_state_content = json.loads(fs.get(ObjectId(node_state_id)).read().decode())

                # Step 4: AI Model Selection (Collapsible)
                with st.expander("🤖 AI Model Configuration", expanded=st.session_state["sections_expanded"]["model"]):
                    provider = st.selectbox("Select Provider:", ["GPT", "Anthropic", "DeepSeek"])
                    model_options = {
                        "GPT": ["gpt-4o", "gpt-3.5-turbo"],
                        "Anthropic": ["claude-3-5-sonnet"],
                        "DeepSeek": ["deepseek-chat"]
                    }
                    model = st.selectbox("Select Model:", model_options[provider])
                    api_key = st.text_input("Enter API Key:", type="password")

                    # 🚀 Validation for "Chat Now" button
                    missing_fields = []
                    if not provider:
                        missing_fields.append("AI Provider")
                    if not model:
                        missing_fields.append("AI Model")
                    if not api_key.strip():
                        missing_fields.append("API Key")

                    # ✅ Display missing field messages
                    if missing_fields:
                        st.warning(f"⚠️ Please provide: {', '.join(missing_fields)}.")

                    # ✅ Enable "Chat Now" button only if all fields are filled
                    chat_ready = not missing_fields
                    chat_now_button = st.button("🚀 Chat Now", disabled=not chat_ready)

                    if chat_now_button:
                        st.session_state["chat_started"] = True  # Store chat state
                        st.session_state["chat_history"] = []  # Reset chat history when chat starts
                        # 🔥 Collapse all sections except chat
                        st.session_state["sections_expanded"] = {"hypothesis": False, "file": False, "model": False, "chat": True}
                        st.rerun()

                # Step 5: Chat Interface (Auto-expanded after "Chat Now")
                with st.expander("💬 Chat with Node", expanded=st.session_state["sections_expanded"]["chat"]):
                    # ✅ Display chat history with original UI colors
                    for role, text in st.session_state["chat_history"]:
                        if role == "user":
                            with st.chat_message(role, avatar="👤"):  # Right-aligned
                                st.markdown(text)
                        else:
                            with st.chat_message(role, avatar="🤖"):  # Left-aligned
                                st.markdown(text)

                    # ✅ User input (Enter sends message)
                    user_input = st.chat_input("Ask a question about the node state:")
                    
                    if user_input:
                        # ✅ Add user question to chat history
                        st.session_state["chat_history"].append(("user", user_input))

                        # ✅ Query LLM
                        response = chat_with_node(api_key, provider, model, node_state_content, user_input)

                        # ✅ Add AI response to chat history
                        st.session_state["chat_history"].append(("assistant", response))

                        # ✅ Refresh chat UI
                        st.rerun()

            else:
                st.warning("⚠️ No stored node state found for this file.")

    else:
        st.warning("⚠️ No processed files found for this Hypothesis ID.")
