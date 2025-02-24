This work and assosciated code is courtesy of the GAIA Lab project.

# InFact: Building Trust in Science through Collaborative Evaluation
*A Gaia Lab project*

<br>

> What we should do is create an institution that collects and evaluates scientific evidence and gives out confidence values based on evidence. -- [Sabine Hossenfelder](https://www.youtube.com/watch?v=zucXnn64qtk&t=314s)

The InFact Project is our attempt to realize this vision. We're building a prototype for a decentralized system that evaluates scientific claims and provides a clear measure of confidence based on available evidence.  Imagine a collaborative platform where scientists and the public can work together to assess the reliability of scientific findings, supported by AI and rigorous automated statistics. This is the core idea behind InFact.

##How does InFact work?

At its heart, InFact uses a network of interconnected nodes. Each node focuses on a specific scientific question, like "Do human-generated greenhouse gas emissions significantly increase global temperatures?"

Within each node, a sophisticated "inference engine" analyzes data related to the question. This engine combines the power of artificial intelligence (specifically, large language models or LLMs) with Bayesian statistics, a mathematical framework for updating beliefs based on evidence.

Breaking down the process:

* Data collection:  The node gathers data from various sources (research papers, datasets, etc.) related to the scientific question.

* AI-powered analysis:  LLMs are used to automatically extract key information from the data, identify relevant studies, and even assess the quality of the evidence.

* Bayesian updating:  The system uses Bayesian methods to weigh the evidence and update a "confidence score" for the scientific claim. This score reflects the strength of the evidence supporting the claim.

* Transparency and traceability:  All data, analyses, and confidence scores are recorded and made available for scrutiny. This ensures transparency and allows for continuous improvement of the system.

## Addressing the Challenges of Data Analysis

One of the biggest challenges in evaluating scientific claims is the sheer diversity and complexity of scientific data. InFact tackles this challenge by using LLMs to generate custom data analysis pipelines for each new piece of evidence. We use frontier off-the-shelf LLMs (currently, Claude 3.5 Sonnet). These AI models are pre-trained on vast amounts of scientific literature, allowing them to adapt to different types of studies and data formats.

## Beyond the Prototype

While our current prototype relies heavily on LLMs, we recognize the need for even greater rigor. Our team is developing a framework for "automatic progressive data analysis." This framework will combine the flexibility of LLMs with the reliability of established statistical models, creating a more robust and trustworthy system for evaluating scientific claims.

## InFact in Action

We envision InFact as a user-friendly platform that presents complex scientific information in a clear and accessible way. Imagine interactive visualizations that show how confidence scores evolve as new evidence emerges, along with explanations that help users understand the reasoning behind the scores.

## The Future of Scientific Confidence

InFact is more than just a technology; it's a vision for a future where scientific knowledge is more accessible, transparent, and trustworthy. By empowering scientists and the public to collaboratively evaluate evidence, we can foster a deeper understanding of science and its role in shaping our world.

## The Gaia Network

InFact is also envisioned as a demonstration of the capabilities of the [Gaia Network Protocol](https://gaia-lab.de), the Gaia Lab's main project. Visit our website to learn more.





