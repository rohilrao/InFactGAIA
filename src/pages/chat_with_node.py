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

# 📌 UI - Chat with Node State
st.title("💬 Chat with Node")

# Step 1: User enters Hypothesis ID
hypothesis_id = st.text_input("Enter Hypothesis ID:")

# Step 2: Fetch processed files
if hypothesis_id:
    processed_files = list(fs.find({"hypothesis_id": hypothesis_id, "status": "processed"}))
    file_options = {str(f["_id"]): f["filename"] for f in processed_files}

    if processed_files:
        selected_file_id = st.selectbox("Select a processed file:", list(file_options.keys()), format_func=lambda x: file_options[x])

        # Step 3: Load Node State
        if selected_file_id:
            file_meta = next(f for f in processed_files if str(f["_id"]) == selected_file_id)
            node_state_id = file_meta.get("node_state_file_id")

            if node_state_id and fs.exists(ObjectId(node_state_id)):
                node_state_content = json.loads(fs.get(ObjectId(node_state_id)).read().decode())

                # Step 4: AI Model Selection
                st.subheader("🤖 Select AI Model")
                provider = st.selectbox("Select Provider:", ["GPT", "Anthropic", "DeepSeek"])
                model_options = {
                    "GPT": ["gpt-4o", "gpt-3.5-turbo"],
                    "Anthropic": ["claude-3-5-sonnet"],
                    "DeepSeek": ["deepseek-chat"]
                }
                model = st.selectbox("Select Model:", model_options[provider])
                api_key = st.text_input("Enter API Key:", type="password")

                # Step 5: Chat Interface
                if api_key:
                    st.subheader("💬 Chat with Node")
                    user_input = st.text_area("Ask a question about the node state:")

                    if st.button("Send"):
                        if user_input.strip():
                            # Step 6: Call LLM with node state context
                            response = chat_with_node(api_key, provider, model, node_state_content, user_input)
                            st.write(f"🤖 **Node:** {response}")
                        else:
                            st.warning("Please enter a question.")

            else:
                st.warning("⚠️ No stored node state found for this file.")

    else:
        st.warning("⚠️ No processed files found for this Hypothesis ID.")


def chat_with_node(api_key, provider, model, node_state_content, user_input):
    """
    Queries the LLM (GPT, Claude, or DeepSeek) with the node state and user's question.
    Restricts responses to the node state contents.
    """
    system_prompt = f"""
    You are an AI assistant that interacts only with the provided node state.
    You must answer questions **only based on this JSON data** and refuse any off-topic conversations.
    Node State:
    {json.dumps(node_state_content, indent=2)}
    """

    # OpenAI API (Modify for DeepSeek/Anthropic if needed)
    if provider == "GPT":
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response["choices"][0]["message"]["content"]
    
    # Placeholder for other APIs (DeepSeek, Anthropic)
    return "🛠️ AI model integration for this provider is under development."
