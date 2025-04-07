import streamlit as st
from infact_utils import call_llm

def display_setup_step():
    """
    Handles Step 1: AI Model Configuration
    Returns "next" if user proceeds to next step, "back" if user wants to go back
    """
    st.markdown("### :orange[Step 1: AI Model Configuration]")

    # Pre-select values if they exist in session state
    provider_default = st.session_state.get("provider", "openai")
    
    provider = st.selectbox("Select Provider:", ["openai", "anthropic"], 
                           index=["openai", "anthropic"].index(provider_default))
    
    model_options = {
        "openai": ["gpt-4o", "chatgpt-4o-latest"],
        "anthropic": ["claude-3-5-sonnet"]
    }
    
    model_default = st.session_state.get("model", model_options[provider][0])
    if model_default not in model_options[provider]:
        model_default = model_options[provider][0]
        
    model = st.selectbox("Select Model:", model_options[provider], 
                        index=model_options[provider].index(model_default))
    
    # Only show API key input if not verified
    if not st.session_state.get("credentials_verified", False):
        api_key = st.text_input("Enter API Key:", type="password")
        
        verify_col1, verify_col2 = st.columns([1, 5])
        with verify_col1:
            verify_button = st.button("Verify Credentials", use_container_width=True)
        
        if verify_button:
            if not api_key.strip():
                st.error("Please enter an API key.")
                return None
            
            # Check if this is an admin key
            is_admin = False
            credential_key = api_key
            
            try:
                if "ADMIN_KEY" in st.secrets and api_key == st.secrets["ADMIN_KEY"]:
                    is_admin = True
                    
                    if provider == "openai" and "OPENAI_API_KEY" in st.secrets:
                        credential_key = st.secrets["OPENAI_API_KEY"]
                    elif provider == "anthropic" and "ANTHROPIC_API_KEY" in st.secrets:
                        credential_key = st.secrets["ANTHROPIC_API_KEY"]
                    else:
                        st.error(f"Admin {provider} API key not configured.")
                        return None
            except Exception as e:
                # Show error and debugging information
                st.error(f"Error accessing admin secrets: {str(e)}")
                st.error("Please enter your own API key instead")
                st.write("Available secrets:", list(st.secrets.keys()) if hasattr(st, "secrets") else "None")
                return None
            # Test the API key with a simple call
            try:
                with st.spinner("Verifying API credentials..."):
                    test_prompt = "Reply with 'OK' if you received this message."
                    response = call_llm(provider, credential_key, model, test_prompt)
                
                # Store in session state
                st.session_state["provider"] = provider
                st.session_state["model"] = model
                st.session_state["api_key"] = credential_key
                st.session_state["is_admin"] = is_admin
                st.session_state["credentials_verified"] = True
                
                auth_type = "admin access" if is_admin else "your API key"
                st.success(f"✅ Connected to {provider} ({model}) using {auth_type}")
            except Exception as e:
                st.error(f"Failed to verify credentials: {str(e)}")
                return None
    else:
        # Show current configuration
        auth_type = "admin access" if st.session_state.get("is_admin", False) else "your API key"
        st.success(f"✅ Connected to {st.session_state['provider']} ({st.session_state['model']}) using {auth_type}")
        
        # Allow changing configuration
        if st.button("Change Configuration"):
            st.session_state["credentials_verified"] = False
            st.experimental_rerun()
    
    # Navigation buttons
    nav_col1, nav_col2 = st.columns([1, 5])
    
    with nav_col1:
        if st.button("← Back"):
            return "back"
    
    with nav_col2:
        if st.session_state.get("credentials_verified", False):
            if st.button("Next →"):
                return "next"
    
    return None