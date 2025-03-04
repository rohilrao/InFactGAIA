import streamlit as st

st.set_page_config(page_title="InFactGAIA", page_icon="📂", layout="wide")

# Custom CSS for better spacing and readability
st.markdown("""
    <style>
        .big-title { font-size: 32px; font-weight: bold; }
        .subtitle { font-size: 22px; font-style: italic; }
        .blockquote { font-size: 18px; font-style: italic; color: #555; }
        .blockquote a { color: #1f77b4; text-decoration: none; }
        .section-header { font-size: 24px; font-weight: bold; margin-top: 40px; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-title">InFact: Building Trust in Science through Collaborative Evaluation</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">*A Gaia Lab project*</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Blockquote with Citation
st.markdown("""
    <p class="blockquote">
    > What we should do is create an institution that collects and evaluates scientific evidence and gives out confidence values based on evidence.
    -- <a href="https://www.youtube.com/watch?v=zucXnn64qtk&t=314s" target="_blank">Sabine Hossenfelder</a>
    </p>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
The **InFact Project** is our attempt to realize this vision. We're building a prototype for a decentralized system that evaluates scientific claims and provides a clear measure of confidence based on available evidence.  

Imagine a collaborative platform where scientists and the public can work together to assess the reliability of scientific findings, supported by AI and rigorous automated statistics. This is the core idea behind InFact.
""")

# Section: How does InFact work?
st.markdown('<p class="section-header">How does InFact work?</p>', unsafe_allow_html=True)

st.markdown("""
At its heart, InFact uses a network of interconnected nodes. Each node focuses on a specific scientific question, like:

**"Do human-generated greenhouse gas emissions significantly increase global temperatures?"**

Within each node, a sophisticated **inference engine** analyzes data related to the question. This engine combines the power of **artificial intelligence (AI)** (specifically, large language models or LLMs) with **Bayesian statistics**, a mathematical framework for updating beliefs based on evidence.

### Breaking down the process:
- **📊 Data collection:** The node gathers data from various sources (research papers, datasets, etc.).
- **🤖 AI-powered analysis:** LLMs automatically extract key information, identify relevant studies, and assess evidence quality.
- **⚖️ Bayesian updating:** The system updates a **confidence score** for the scientific claim.
- **🔍 Transparency & traceability:** All data, analyses, and scores are **open and auditable**.
""")

# Section: Addressing the Challenges of Data Analysis
st.markdown('<p class="section-header">Addressing the Challenges of Data Analysis</p>', unsafe_allow_html=True)

st.markdown("""
One of the biggest challenges in evaluating scientific claims is the **diversity and complexity of data**. InFact tackles this challenge by using **LLMs to generate custom data analysis pipelines** for each new piece of evidence.  

We currently use **Claude 3.5 Sonnet**, a cutting-edge AI model pre-trained on vast scientific literature, allowing it to adapt to different study formats.
""")

# Section: Beyond the Prototype
st.markdown('<p class="section-header">Beyond the Prototype</p>', unsafe_allow_html=True)

st.markdown("""
While our current prototype relies heavily on LLMs, we recognize the need for even greater rigor. Our team is developing a framework for **"automatic progressive data analysis"**, combining:
- The **flexibility of LLMs**
- The **reliability of established statistical models**

This will result in a more **trustworthy** system for evaluating scientific claims.
""")

# Section: InFact in Action
st.markdown('<p class="section-header">InFact in Action</p>', unsafe_allow_html=True)

st.markdown("""
We envision InFact as a **user-friendly platform** that presents complex scientific information in an **accessible way**.  

🔹 **Interactive visualizations** show how confidence scores evolve with new evidence.  
🔹 **Clear explanations** help users understand the reasoning behind scores.  
""")

# Section: The Future of Scientific Confidence
st.markdown('<p class="section-header">The Future of Scientific Confidence</p>', unsafe_allow_html=True)

st.markdown("""
InFact is more than just a technology; it's a **vision** for making **scientific knowledge accessible, transparent, and trustworthy**.  

By empowering both **scientists and the public** to collaboratively evaluate evidence, we aim to foster **a deeper understanding of science** and its role in shaping our world.
""")

# Section: The Gaia Network
st.markdown('<p class="section-header">The Gaia Network</p>', unsafe_allow_html=True)

st.markdown("""
InFact is also envisioned as a **demonstration of the capabilities of the** [**Gaia Network Protocol**](https://gaia-lab.de), the **Gaia Lab's main project**.  
Visit our website to learn more! 🌍
""")
