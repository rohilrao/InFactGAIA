import streamlit as st 

st.set_page_config(page_title="GAIA - InFact - Demo", page_icon="", layout="wide")

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

Imagine a collaborative platform where scientists and the public can <b>work together</b> to assess the reliability of scientific findings, 
supported by <b>AI-powered analysis</b> and <b>rigorous automated statistics</b>.  
This is the core idea behind InFact.
""", unsafe_allow_html=True)

# Section: How does InFact work?
st.markdown("### :orange[How does InFact work?]")
st.markdown("""
At its heart, InFact uses a <b>network of interconnected nodes</b>. Each node focuses on a specific scientific question, like 
"Do human-generated greenhouse gas emissions significantly increase global temperatures?"

Within each node, a sophisticated <b>"inference engine"</b> analyzes data related to the question. This engine combines the power of 
artificial intelligence (specifically, large language models or LLMs) with Bayesian statistics, a mathematical framework for 
updating beliefs based on evidence.

<b>Breaking down the process:</b>

• <b>Data collection:</b> The node gathers data from various sources (research papers, datasets, etc.) related to the scientific question.

• <b>AI-powered analysis:</b> LLMs are used to automatically extract key information from the data, identify relevant studies, 
and even assess the quality of the evidence.

• <b>Bayesian updating:</b> The system uses Bayesian methods to weigh the evidence and update a "confidence score" for the scientific claim. 
This score reflects the strength of the evidence supporting the claim.

• <b>Transparency and traceability:</b> All data, analyses, and confidence scores are recorded and made available for scrutiny. 
This ensures transparency and allows for continuous improvement of the system.
""", unsafe_allow_html=True)

# Section: Addressing the Challenges of Data Analysis
st.markdown("### :orange[Addressing the Challenges of Data Analysis]")
st.markdown("""
One of the biggest challenges in evaluating scientific claims is the <b>sheer diversity and complexity of scientific data</b>. 
InFact tackles this challenge by using <b>LLMs to generate custom data analysis pipelines</b> for each new piece of evidence. 
We use frontier off-the-shelf LLMs (currently, <b>Claude 3.5 Sonnet</b>). These AI models are pre-trained on vast amounts of 
scientific literature, allowing them to adapt to different types of studies and data formats.
""", unsafe_allow_html=True)

# Section: Beyond the Prototype
st.markdown("### :orange[Beyond the Prototype]")
st.markdown("""
While our current prototype relies heavily on LLMs, we recognize the need for even greater rigor. Our team is developing a 
framework for <b>"automatic progressive data analysis."</b> This framework will combine the flexibility of LLMs with the 
reliability of established statistical models, creating a <b>more robust and trustworthy system</b> for evaluating scientific claims.
""", unsafe_allow_html=True)

# Section: InFact in Action
st.markdown("### :orange[InFact in Action]")
st.markdown("""
We envision InFact as a <b>user-friendly platform</b> that presents complex scientific information in a <b>clear and accessible way</b>. 
Imagine interactive visualizations that show how confidence scores evolve as new evidence emerges, along with explanations that 
help users understand the reasoning behind the scores.
""", unsafe_allow_html=True)

# Section: The Future of Scientific Confidence
st.markdown("### :orange[The Future of Scientific Confidence]")
st.markdown("""
<b>InFact is more than just a technology; it's a vision for a future where scientific knowledge is more accessible, transparent, 
and trustworthy.</b> By empowering scientists and the public to collaboratively evaluate evidence, we can foster a deeper 
understanding of science and its role in shaping our world.
""", unsafe_allow_html=True)

# Section: The Gaia Network
st.markdown("### :orange[The Gaia Network]")
st.markdown("""
InFact is also envisioned as a <b>demonstration of the capabilities of the</b> 
<a href="https://gaia-lab.de"><b>Gaia Network Protocol</b></a>, the <b>Gaia Lab's main project</b>.  
Visit our website to learn more!
""", unsafe_allow_html=True)