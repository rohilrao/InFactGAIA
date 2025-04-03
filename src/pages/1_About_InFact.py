import streamlit as st 

st.set_page_config(page_title="InFactGAIAV3", page_icon="📂", layout="wide")

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
that evaluates scientific claims and provides a clear measure of confidence based on available evidence.  
""", unsafe_allow_html=True)

# Section: How does InFact work?
st.markdown("### :orange[How does InFact work?]")
st.markdown("""
At its heart, InFact uses a <b>network of interconnected nodes</b>. Each node focuses on a specific scientific question, such as:

> <b>Do human-generated greenhouse gas emissions significantly increase global temperatures?</b>

Within each node, an <b>inference engine</b> analyzes data related to the question. This system combines:
- <b>AI-powered evidence extraction</b>
- <b>Bayesian statistical models</b>
- <b>Fully auditable confidence scoring</b>
""", unsafe_allow_html=True)

# Section: Challenges in Scientific Evaluation
st.markdown("### :orange[Challenges in Scientific Evaluation]")
st.markdown("""
One of the biggest challenges in evaluating scientific claims is the <b>diversity and complexity of data</b>.  
InFact tackles this challenge using <b>LLMs to generate adaptive data analysis pipelines</b>.  

We currently integrate <b>Claude 3.5 Sonnet</b>, a state-of-the-art AI model trained on scientific literature, 
allowing it to analyze different study formats with precision.
""", unsafe_allow_html=True)

# Section: Beyond the Prototype
st.markdown("### :orange[Beyond the Prototype]")
st.markdown("""
To address this, our team is developing a <b>hybrid analysis framework</b> that blends:
- <b>The adaptability of AI</b>
- <b>The reliability of established statistical methods</b>

This will result in a <b>robust, bias-resistant system</b> for evaluating scientific claims.
""", unsafe_allow_html=True)

# Section: InFact in Action
st.markdown("### :orange[InFact in Action]")
st.markdown("""
Our goal is to build a <b>user-friendly platform</b> that presents complex scientific evidence in an <b>accessible format</b>.

🔹 <b>Interactive visualizations</b> track how confidence scores evolve over time.  
🔹 <b>Transparent explanations</b> clarify why specific conclusions are reached.  
""", unsafe_allow_html=True)

# Section: The Future of Scientific Confidence
st.markdown("### :orange[The Future of Scientific Confidence]")
st.markdown("""
<b>InFact is not just a tool—it's a movement towards a more transparent scientific process.</b>  
By allowing <b>scientists and the public</b> to work together in assessing claims, 
we aim to <b>increase public trust in science and improve evidence-based decision-making</b>.
""", unsafe_allow_html=True)

# Section: The Gaia Network
st.markdown("### :orange[The Gaia Network]")
st.markdown("""
InFact is also envisioned as a <b>demonstration of the capabilities of the</b> 
<a href="https://gaia-lab.de"><b>Gaia Network Protocol</b></a>, the <b>Gaia Lab's main project</b>.  
Visit our website to learn more!
""", unsafe_allow_html=True)