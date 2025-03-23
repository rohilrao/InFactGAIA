import streamlit as st

def step_1_model_config():
    st.header("Step 1: AI Model Configuration")

    provider = st.selectbox("Select Provider:", ["GPT", "Anthropic"])
    model_options = {
        "GPT": ["gpt-4o", "chatgpt-4o-latest"],
        "Anthropic": ["claude-3-5-sonnet"]
    }
    model = st.selectbox("Select Model:", model_options[provider])
    api_key = st.text_input("Enter API Key:", type="password")

    if st.button("Next →"):
        if not provider or not model or not api_key.strip():
            st.warning("Please fill in all fields before proceeding.")
        else:
            st.session_state["provider"] = provider
            st.session_state["model"] = model
            st.session_state["api_key"] = api_key
            st.session_state.process_step = 2
            st.rerun()
