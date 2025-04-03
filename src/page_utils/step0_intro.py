import streamlit as st
import sys
import os

def display_intro():
    """
    Display the introduction page for InFact Demo with a next button.
    
    Returns:
        bool: True if the user clicked Next, False otherwise
    """
    # Title & Subtitle
    st.markdown("### :orange[InFact Demo: Building Trust in Science through Collaborative Evaluation]")
    st.markdown("#### :orange[A Gaia Lab Project]")
    
    # 📌 Blockquote
    st.markdown("""
        <p class="blockquote">
        "What we should do is create an institution that collects and evaluates scientific evidence and gives out confidence values based on evidence."
        <br>— <a href="https://www.youtube.com/watch?v=zucXnn64qtk&t=314s" target="_blank">Sabine Hossenfelder</a>
        </p>
    """, unsafe_allow_html=True)
    
    # Section: Introduction
    st.markdown("### :orange[Introduction]")
    st.markdown("""
    The <b>InFact Project</b> is our attempt to realize this vision. We're building a prototype for a decentralized system 
    that evaluates scientific claims and provides a clear measure of confidence based on available evidence. Imagine a collaborative platform where scientists and the public can <b>work together</b> to assess the reliability of scientific findings, 
    supported by <b>AI-powered analysis</b> and <b>rigorous automated statistics</b>.  
    This is the core idea behind InFact.
    """, unsafe_allow_html=True)
    
    # Section: The Gaia Network
    st.markdown("### :orange[The Gaia Network]")
    st.markdown("""
    InFact is also envisioned as a <b>demonstration of the capabilities of the</b> 
    <a href="https://gaia-lab.de"><b>Gaia Network Protocol</b></a>, the <b>Gaia Lab's main project</b>.  
    Visit our website to learn more!
    """, unsafe_allow_html=True)
    
    # Add a spacer
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Add Next button at the bottom right
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        next_clicked = st.button("Next →", use_container_width=True)
    
    return next_clicked