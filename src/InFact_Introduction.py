import streamlit as st 

st.set_page_config(page_title="InFactGAIAV3", page_icon="📂", layout="wide")

# Custom CSS for improved readability in **dark mode**
st.markdown("""
    <style>
        /* Title & Headers */
        .big-title { font-size: 32px; font-weight: bold; margin-bottom: 5px; color: #ffffff; }
        .subtitle { font-size: 22px; font-style: italic; color: #cccccc; }
        .section-header { 
            font-size: 24px; 
            font-weight: bold; 
            margin-top: 40px; 
            padding-bottom: 5px; 
            border-bottom: 2px solid rgba(255, 255, 255, 0.2); 
            color: #ffffff;
        }

        /* Quote Styling for Dark Mode */
        .blockquote { 
            font-size: 18px; 
            font-style: italic; 
            color: #f1f1f1;  /* Lighter text for contrast */
            border-left: 4px solid #1f77b4; 
            padding-left: 15px; 
            margin: 20px 0;
            background-color: rgba(255, 255, 255, 0.1);  /* Subtle contrast for dark mode */
            padding: 10px;
            border-radius: 5px;
        }
        .blockquote a { color: #1f77b4; text-decoration: none; }

        /* Content Box (for sections with extra info) */
        .content-box {
            background-color: rgba(255, 255, 255, 0.05); /* Slightly lighter dark mode */
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 20px;
            color: #dddddd;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<h1 class="big-title">InFact Demo: Building Trust in Science through Collaborative Evaluation</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">A Gaia Lab Project</h2>', unsafe_allow_html=True)



# 📌 **Blockquote with Proper Styling (Dark Mode Optimized)**
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
The <b>InFact Project</b> is our attempt to realize this vision. We're building a prototype for a decentralized system 
that evaluates scientific claims and provides a clear measure of confidence based on available evidence.  

Imagine a collaborative platform where scientists and the public can <b>work together</b> to assess the reliability of scientific findings, 
supported by <b>AI-powered analysis</b> and <b>rigorous automated statistics</b>.  
This is the core idea behind InFact.
</div>
""", unsafe_allow_html=True)

# Section: How does InFact work?
st.markdown('<p class="section-header">How does InFact work?</p>', unsafe_allow_html=True)

st.markdown("""
At its heart, InFact uses a <b>network of interconnected nodes</b>. Each node focuses on a specific scientific question, such as:

> <b>Do human-generated greenhouse gas emissions significantly increase global temperatures?</b>

Within each node, an <b>inference engine</b> analyzes data related to the question. This system combines:
- <b>AI-powered evidence extraction</b>
- <b>Bayesian statistical models</b>
- <b>Fully auditable confidence scoring</b>
""", unsafe_allow_html=True)

# Section: Challenges in Scientific Evaluation
st.markdown('<p class="section-header">Challenges in Scientific Evaluation</p>', unsafe_allow_html=True)

st.markdown("""
<div class="content-box">
One of the biggest challenges in evaluating scientific claims is the <b>diversity and complexity of data</b>.  
InFact tackles this challenge using <b>LLMs to generate adaptive data analysis pipelines</b>.  

We currently integrate <b>Claude 3.5 Sonnet</b>, a state-of-the-art AI model trained on scientific literature, 
allowing it to analyze different study formats with precision.
</div>
""", unsafe_allow_html=True)

# Section: Beyond the Prototype
st.markdown('<p class="section-header">Beyond the Prototype</p>', unsafe_allow_html=True)

st.markdown("""
To address this, our team is developing a <b>hybrid analysis framework</b> that blends:
- <b>The adaptability of AI</b>
- <b>The reliability of established statistical methods</b>

This will result in a <b>robust, bias-resistant system</b> for evaluating scientific claims.
""", unsafe_allow_html=True)

# Section: InFact in Action
st.markdown('<p class="section-header">InFact in Action</p>', unsafe_allow_html=True)

st.markdown("""
Our goal is to build a <b>user-friendly platform</b> that presents complex scientific evidence in an <b>accessible format</b>.

🔹 <b>Interactive visualizations</b> track how confidence scores evolve over time.  
🔹 <b>Transparent explanations</b> clarify why specific conclusions are reached.  
""", unsafe_allow_html=True)

# Section: The Future of Scientific Confidence
st.markdown('<p class="section-header">The Future of Scientific Confidence</p>', unsafe_allow_html=True)

st.markdown("""
<b>InFact is not just a tool—it's a movement towards a more transparent scientific process.</b>  
By allowing <b>scientists and the public</b> to work together in assessing claims, 
we aim to <b>increase public trust in science and improve evidence-based decision-making</b>.
""", unsafe_allow_html=True)

# Section: The Gaia Network
st.markdown('<p class="section-header">The Gaia Network</p>', unsafe_allow_html=True)
st.markdown("""
InFact is also envisioned as a <b>demonstration of the capabilities of the</b> 
<a href="https://gaia-lab.de"><b>Gaia Network Protocol</b></a>, the <b>Gaia Lab's main project</b>.  
Visit our website to learn more!
""", unsafe_allow_html=True)