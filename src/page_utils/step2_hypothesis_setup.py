import streamlit as st
import sys
import os

# Add the necessary paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import necessary modules
try:
    from InFact.providers.anthropic_provider import AnthropicProvider
    from InFact.providers.openai_provider import OpenAIProvider
    from hypothesis_setup_utils import initialize_session_state
    from hypothesis_setup_utils import setup_ui_styles
    from hypothesis_setup_utils import render_hypothesis_setup
    from hypothesis_setup_utils import render_hypothesis_refinement
    from hypothesis_setup_utils import render_description_section
    from hypothesis_setup_utils import render_background_section
    from hypothesis_setup_utils import render_chat_section
    from hypothesis_setup_utils import render_current_state
    from hypothesis_setup_utils import render_navigation_buttons
    from hypothesis_setup_utils import check_hypothesis_id

except ImportError as e:
    st.error(f"Could not import required modules: {str(e)}. Please ensure the packages are installed correctly.")

def display_combined_hypothesis_step(hypothesis_collection, call_llm=None):
    """
    Main function for the hypothesis page. Calls the modular components in sequence.
    
    Args:
        hypothesis_collection: MongoDB collection for hypotheses
        call_llm: Function to call LLM API (optional, for external LLM calls)
        
    Returns:
        str: Navigation action - "back", "next", or None
    """
    # Initialize session state variables
    initialize_session_state()
    
    # Set up UI styles
    setup_ui_styles()
    
    # Render hypothesis setup section
    hypothesis_id, hypothesis_doc = render_hypothesis_setup(hypothesis_collection)
    
    # Explicitly set the hypothesis_id in session state
    if hypothesis_id:
        print(f"[display_combined_hypothesis_step] Setting hypothesis_id in session state: {hypothesis_id}")
        st.session_state["hypothesis_id"] = hypothesis_id
    else:   
        print(f"[display_combined_hypothesis_step] No hypothesis_id provided.")

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
    
    # Get the active hypothesis ID and document
    active_id = st.session_state.get("hypothesis_id", hypothesis_id)
    if not hypothesis_doc:
        hypothesis_doc = hypothesis_collection.find_one({"_id": active_id})
    
    if not hypothesis_doc:
        st.error("Could not retrieve hypothesis data.")
        return None
    
    # Make sure the hypothesis ID status is up to date
    check_hypothesis_id(hypothesis_collection)
    
    # Render hypothesis refinement section
    updated_doc = render_hypothesis_refinement(active_id, hypothesis_doc, hypothesis_collection, call_llm)
    if updated_doc:
        hypothesis_doc = updated_doc
    
    # Render description section
    updated_doc = render_description_section(active_id, hypothesis_doc, hypothesis_collection, call_llm)
    if updated_doc:
        hypothesis_doc = updated_doc
    
    # Render background section
    updated_doc = render_background_section(active_id, hypothesis_doc, hypothesis_collection, call_llm)
    if updated_doc:
        hypothesis_doc = updated_doc
    
    # Render chat section
    render_chat_section(active_id, hypothesis_doc, hypothesis_collection, call_llm)
    
    # Render current state
    #render_current_state(active_id)
    
    # Render navigation buttons
    return render_navigation_buttons(active_id, hypothesis_doc)