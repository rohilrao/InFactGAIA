import streamlit as st 

st.set_page_config(page_title="InFactGAIA", page_icon="📂", layout="wide")

# Custom CSS for improved readability & spacing
st.markdown("""
    <style>
        .big-title { font-size: 32px; font-weight: bold; margin-bottom: 5px; }
        .subtitle { font-size: 22px; font-style: italic; color: #555; }
        .blockquote { 
            font-size: 18px; 
            font-style: italic; 
            color: #333; 
            border-left: 4px solid #1f77b4; 
            padding-left: 15px; 
            margin: 20px 0;
        }
        .section-header { 
            font-size: 24px; 
            font-weight: bold; 
            margin-top: 40px; 
            padding-bottom: 5px; 
            border-bottom: 2px solid #ddd;
        }
        .content-box {
            background-color: #f9f9f9; 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<h1 class="big-title">InFact Demo: Building Trust in Science through Collaborative Evaluation</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">A Gaia Lab Project</h2>', unsafe_allow_html=True)

# Section: The Gaia Network
st.markdown('<p class="section-header">The Gaia Network</p>', unsafe_allow_html=True)
st.markdown("""
InFact is also envisioned as a **demonstration of the capabilities of the** 
[**Gaia Network Protocol**](https://gaia-lab.de), the **Gaia Lab's main project**.  
Visit our website to learn more!
""")

# 📌 **Blockquote with Proper Styling**
st.markdown("""
    <p class="blockquote">
    "What we should do is create an institution that collects and evaluates scientific evidence and gives out confidence values based on evidence."
    <br>— <a href="https://www.youtube.com/watch?v=zucXnn64qtk&t=314s" target="_blank">Sabine Hossenfelder</a>
    </p>
""", unsafe_allow_html=True)

# Introduction
st.markdown('<p class="section-header">Introduction</p>', unsafe_allow_html=True)
st.markdown("""
<div class="content-box">
The **InFact Project** is our attempt to realize this vision. We're building a prototype for a decentralized system 
that evaluates scientific claims and provides a clear measure of confidence based on available evidence.  

Imagine a collaborative platform where scientists and the public can **work together** to assess the reliability of scientific findings, 
supported by **AI-powered analysis** and **rigorous automated statistics**.  
This is the core idea behind InFact.
</div>
""", unsafe_allow_html=True)

# Section: How does InFact work?
st.markdown('<p class="section-header">How does InFact work?</p>', unsafe_allow_html=True)

st.markdown("""
At its heart, InFact uses a **network of interconnected nodes**. Each node focuses on a specific scientific question, such as:

> **"Do human-generated greenhouse gas emissions significantly increase global temperatures?"**

Within each node, an **inference engine** analyzes data related to the question. This system combines:
- **AI-powered evidence extraction**
- **Bayesian statistical models**
- **Fully auditable confidence scoring**
""")

# Section: Challenges in Scientific Evaluation
st.markdown('<p class="section-header">Challenges in Scientific Evaluation</p>', unsafe_allow_html=True)

st.markdown("""
<div class="content-box">
One of the biggest challenges in evaluating scientific claims is the **diversity and complexity of data**.  
InFact tackles this challenge using **LLMs to generate adaptive data analysis pipelines**.  

We currently integrate **Claude 3.5 Sonnet**, a state-of-the-art AI model trained on scientific literature, 
allowing it to analyze different study formats with precision.
</div>
""", unsafe_allow_html=True)

# Section: Beyond the Prototype
st.markdown('<p class="section-header">Beyond the Prototype</p>', unsafe_allow_html=True)

st.markdown("""
We recognize the need for even greater **scientific rigor**.  
To address this, our team is developing a **hybrid analysis framework** that blends:
✅ **The adaptability of AI**  
✅ **The reliability of established statistical methods**  
This will result in a **robust, bias-resistant system** for evaluating scientific claims.
""")

# Section: InFact in Action
st.markdown('<p class="section-header">InFact in Action</p>', unsafe_allow_html=True)

st.markdown("""
Our goal is to build a **user-friendly platform** that presents complex scientific evidence in an **accessible format**.

🔹 **Interactive visualizations** track how confidence scores evolve over time.  
🔹 **Transparent explanations** clarify why specific conclusions are reached.  
""")

# Section: The Future of Scientific Confidence
st.markdown('<p class="section-header">The Future of Scientific Confidence</p>', unsafe_allow_html=True)

st.markdown("""
**InFact is not just a tool—it's a movement towards a more transparent scientific process.**  
By allowing **scientists and the public** to work together in assessing claims, 
we aim to **increase public trust in science and improve evidence-based decision-making**.
""")
