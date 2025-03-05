import streamlit as st
from pymongo import MongoClient
import gridfs
import json
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi
import openai  # Or Anthropic API, DeepSeek API, etc.


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

        client = OpenAI(api_key=api_key)  # ✅ Initialize OpenAI Client
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content  # ✅ Fix response structure

    # Placeholder for other APIs (DeepSeek, Anthropic)
    return "🛠️ AI model integration for this provider is under development."



import streamlit as st
from pymongo import MongoClient
import gridfs
import json
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi
import openai  # Or Anthropic API, DeepSeek API, etc.

# 🔐 MongoDB Connection
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["hypothesis_management"]
fs = gridfs.GridFS(db)

# 📌 UI - Chat with Node
st.title("💬 Chat with Node")

# Step 1: User enters Hypothesis ID (Collapsible)
with st.expander("📌 Enter Hypothesis ID"):
    hypothesis_id = st.text_input("Enter Hypothesis ID:")
    if st.button("🔍 Load Processed Files"):
        st.session_state["loaded_hypothesis_id"] = hypothesis_id  # Store ID in session

# Get stored Hypothesis ID after button click
loaded_hypothesis_id = st.session_state.get("loaded_hypothesis_id", None)

if loaded_hypothesis_id:
    processed_files = list(fs.find({"hypothesis_id": loaded_hypothesis_id, "status": "processed"}))
    file_options = {str(f._id): f.filename for f in processed_files}

    if processed_files:
        # Step 2: Select a Processed File (Collapsible)
        with st.expander("📂 Select a Processed File"):
            selected_file_id = st.selectbox("Select a file:", list(file_options.keys()), format_func=lambda x: file_options[x])

            if st.button("📥 Load Node State"):
                st.session_state["selected_file_id"] = selected_file_id  # Store file ID in session

        # Step 3: Load Node State
        selected_file_id = st.session_state.get("selected_file_id", None)

        if selected_file_id:
            file_meta = next(f for f in processed_files if str(f._id) == selected_file_id)
            node_state_id = getattr(file_meta, "node_state_file_id", None)

            # ✅ Corrected way to check if `node_state_id` exists in GridFS
            if node_state_id and fs.find_one({"_id": ObjectId(node_state_id)}):
                node_state_content = json.loads(fs.get(ObjectId(node_state_id)).read().decode())

                # Step 4: AI Model Selection (Collapsible)
                with st.expander("🤖 AI Model Configuration"):
                    provider = st.selectbox("Select Provider:", ["GPT", "Anthropic", "DeepSeek"])
                    model_options = {
                        "GPT": ["gpt-4o", "gpt-3.5-turbo"],
                        "Anthropic": ["claude-3-5-sonnet"],
                        "DeepSeek": ["deepseek-chat"]
                    }
                    model = st.selectbox("Select Model:", model_options[provider])
                    api_key = st.text_input("Enter API Key:", type="password")

                    # Button to Start Chat
                    if st.button("🚀 Chat Now"):
                        st.session_state["chat_started"] = True  # Store chat state

                # Step 5: Chat Interface (Visible only after clicking "Chat Now")
                if st.session_state.get("chat_started", False):
                    st.subheader("💬 Chat with Node")
                    user_input = st.text_area("Ask a question about the node state:")

                    if st.button("✉️ Send Question"):
                        if user_input.strip():
                            response = chat_with_node(api_key, provider, model, node_state_content, user_input)
                            st.write(f"🤖 **Node:** {response}")
                        else:
                            st.warning("⚠️ Please enter a question.")
            else:
                st.warning("⚠️ No stored node state found for this file.")

    else:
        st.warning("⚠️ No processed files found for this Hypothesis ID.")

# 🔹 Chat Function (Same as before)
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
