import streamlit as st

def display_setup_step():
    """
    Handles Step 1: AI Model Configuration
    Returns True if user proceeds to next step, False otherwise
    """
    st.markdown("### :orange[Step 1: AI Model Configuration]")
    #st.markdown("### :orange[Upload Files for Your Hypothesis]")

    provider = st.selectbox("Select Provider:", ["openai", "anthropic"])  # Only GPT or Anthropic
    model_options = {
        "openai": ["gpt-4o", "chatgpt-4o-latest"],   # Example GPT models
        "anthropic": ["claude-3-5-sonnet"]   # Example Anthropic model
    }
    model = st.selectbox("Select Model:", model_options[provider])
    api_key = st.text_input("Enter API Key:", type="password")

    proceed = False
    if st.button("Next →"):
        if not provider or not model or not api_key.strip():
            st.warning("Please fill in all fields before proceeding.")
        else:
            # Secret admin key check
            # First check if the provided key is exactly "infactadmin"
            if api_key == "infactadmin":
                # Get the appropriate API key based on selected provider
                if provider == "openai":
                    # Try to get from Streamlit secrets
                    if "OPENAI_API_KEY" in st.secrets:
                        api_key = st.secrets["OPENAI_API_KEY"]
                    else:
                        st.warning("Admin OpenAI API key not configured in secrets. Using provided key instead.")
                
                elif provider == "anthropic":
                    # Try to get from Streamlit secrets
                    if "ANTHROPIC_API_KEY" in st.secrets:
                        api_key = st.secrets["ANTHROPIC_API_KEY"]
                    else:
                        st.warning("Admin Anthropic API key not configured in secrets. Using provided key instead.")
            
            # Also check if there's a custom admin key defined in secrets
            elif "ADMIN_KEY" in st.secrets and api_key == st.secrets["ADMIN_KEY"]:
                # Get the appropriate API key based on selected provider
                if provider == "openai":
                    # Try to get from Streamlit secrets
                    if "OPENAI_API_KEY" in st.secrets:
                        api_key = st.secrets["OPENAI_API_KEY"]
                    else:
                        st.warning("Admin OpenAI API key not configured in secrets. Using provided key instead.")
                
                elif provider == "anthropic":
                    # Try to get from Streamlit secrets
                    if "ANTHROPIC_API_KEY" in st.secrets:
                        api_key = st.secrets["ANTHROPIC_API_KEY"]
                    else:
                        st.warning("Admin Anthropic API key not configured in secrets. Using provided key instead.")
            
            # Save final values to session state
            st.session_state["provider"] = provider
            st.session_state["model"] = model
            st.session_state["api_key"] = api_key
            proceed = True
    
    return proceed