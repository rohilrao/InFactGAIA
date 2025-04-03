import streamlit as st



def display_setup_step():
    """
    Handles Step 1: AI Model Configuration
    Returns True if user proceeds to next step, False otherwise
    Returns -1 if user wants to go back
    """
    st.markdown("### :orange[Step 1: AI Model Configuration]")
    #st.markdown("### :orange[Upload Files for Your Hypothesis]")

    # Pre-select values if they exist in session state
    provider_default = st.session_state.get("provider", "openai")
    
    provider = st.selectbox("Select Provider:", ["openai", "anthropic"], index=["openai", "anthropic"].index(provider_default))
    
    model_options = {
        "openai": ["gpt-4o", "chatgpt-4o-latest"],   # Example GPT models
        "anthropic": ["claude-3-5-sonnet"]   # Example Anthropic model
    }
    
    model_default = st.session_state.get("model", model_options[provider][0])
    # Ensure the default model is valid for the selected provider
    if model_default not in model_options[provider]:
        model_default = model_options[provider][0]
        
    model = st.selectbox("Select Model:", model_options[provider], 
                        index=model_options[provider].index(model_default))
    
    # Don't default to showing the API key when returning
    api_key = st.text_input("Enter API Key:", type="password")

    # Initialize credentials_verified in session state if it doesn't exist
    if "credentials_verified" not in st.session_state:
        st.session_state["credentials_verified"] = False
        
    # Add a Set Credentials button
    if st.button("Set Credentials"):
        if not provider or not model or not api_key.strip():
            st.warning("Please fill in all fields before setting credentials.")
            st.session_state["credentials_verified"] = False
        else:
            # Always evaluate credentials fresh each time
            credential_key = api_key
            is_admin = False
            
            print(st.secrets)

            # Check if this is an admin key
            if "ADMIN_KEY" in st.secrets:
                is_admin = (api_key == st.secrets["ADMIN_KEY"])
                
                # If admin access granted, use the appropriate API key from secrets
                if is_admin:
                    if provider == "openai":
                        if "OPENAI_API_KEY" in st.secrets:
                            credential_key = st.secrets["OPENAI_API_KEY"]
                            st.success("Using admin OpenAI API key")
                        else:
                            st.warning("Admin OpenAI API key not configured in secrets.")
                            is_admin = False  # Revert admin status since we can't use admin credentials
                    
                    elif provider == "anthropic":
                        if "ANTHROPIC_API_KEY" in st.secrets:
                            credential_key = st.secrets["ANTHROPIC_API_KEY"]
                            st.success("Using admin Anthropic API key")
                        else:
                            st.warning("Admin Anthropic API key not configured in secrets.")
                            is_admin = False  # Revert admin status since we can't use admin credentials
                else:
                    # Only show this message if the key provided doesn't match admin key
                    st.info("Using your provided API key directly.")
            else:
                st.error("ADMIN_KEY not found in Streamlit secrets. Admin access unavailable.")
                st.info("Using your provided API key directly.")
            
            # Save final values to session state
            st.session_state["provider"] = provider
            st.session_state["model"] = model
            st.session_state["api_key"] = credential_key
            st.session_state["is_admin"] = is_admin  # Track admin status
            st.session_state["credentials_verified"] = True
            
            if not is_admin:
                st.success("Credentials set successfully!")
    
    # Show status based on credentials verification
    if st.session_state["credentials_verified"]:
        auth_method = "admin credentials" if st.session_state.get("is_admin", False) else "your API key"
        st.info(f"Ready to proceed with {provider} ({model}) using {auth_method}")
    else:
        st.warning("Please set your credentials before proceeding")
    
    # Create columns for Back and Next buttons
    col1, col2 = st.columns([1, 5])
    
    # Back button in the first column
    with col1:
        if st.button("← Back"):
            return -1
    
    # Only show the Next button if credentials are verified
    proceed = False
    with col2:
        if st.session_state["credentials_verified"]:
            if st.button("Next →"):
                proceed = True
    
    return proceed