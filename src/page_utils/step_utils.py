import streamlit as st

process_steps = {
    1: "Setup",
    2: "Create Hypothesis", 
    3: "Generate Summary",
    4: "Upload Files",
    5: "Interactive Code Review",
    6: "Evidence Processing",
    7: "View Results"
}

def initialize_session_state():
    """Initialize necessary session state variables if they don't exist."""
    if "process_step" not in st.session_state:
        st.session_state.process_step = 1

def show_step_progress():
    """Display step progress indicator at the top of the app."""
    current_step = st.session_state.process_step
    
    # Create a container for the step progress
    step_container = st.container()
    
    with step_container:
        cols = st.columns(len(process_steps))
        
        for i, (step_num, step_name) in enumerate(process_steps.items()):
            with cols[i]:
                if step_num < current_step:
                    # Completed step
                    st.markdown(f"<div style='text-align: center; color: green;'>✓<br>{step_name}</div>", unsafe_allow_html=True)
                elif step_num == current_step:
                    # Current step
                    st.markdown(f"<div style='text-align: center; font-weight: bold;'>→<br>{step_name}</div>", unsafe_allow_html=True)
                else:
                    # Future step
                    st.markdown(f"<div style='text-align: center; color: gray;'>{step_num}<br>{step_name}</div>", unsafe_allow_html=True)
    
    # Add a separator
    st.markdown("---")