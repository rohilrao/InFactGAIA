import streamlit as st

def display_setup_step():
    """
    Handles Step 1: AI Model Configuration
    Returns True if user proceeds to next step, False otherwise
    """
    st.markdown("### :orange[Step 1: AI Model Configuration]")
    #st.markdown("### :orange[Upload Files for Your Hypothesis]")

    provider = st.selectbox("Select Provider:", ["GPT", "Anthropic"])  # Only GPT or Anthropic
    model_options = {
        "GPT": ["gpt-4o", "chatgpt-4o-latest"],   # Example GPT models
        "Anthropic": ["claude-3-5-sonnet"]   # Example Anthropic model
    }
    model = st.selectbox("Select Model:", model_options[provider])
    api_key = st.text_input("Enter API Key:", type="password")

    proceed = False
    if st.button("Next →"):
        if not provider or not model or not api_key.strip():
            st.warning("Please fill in all fields before proceeding.")
        else:
            st.session_state["provider"] = provider
            st.session_state["model"] = model
            st.session_state["api_key"] = api_key
            proceed = True
    
    return proceed