import streamlit as st
import requests
import time
import random
import xml.etree.ElementTree as ET
from pymongo import MongoClient
from pymongo.server_api import ServerApi


# 📌 **Evidence Fetching Functions**
def search_google(api_key, query, top_n=5):
    """
    Fetches search results from SerpAPI (Google) based on query.
    """
    url = f"{BASE_URL_GOOGLE}?engine=google&q={query}&api_key={api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        st.error("Google API request failed.")
        return []

    data = response.json()
    urls = [(result["title"], result["link"]) for result in data.get("organic_results", [])]
    return urls[:top_n]

def search_semantic_papers(query, limit=5):
    """
    Fetches research papers from Semantic Scholar.
    """
    url = f"{BASE_URL_SEMANTIC}/paper/search"
    params = {"query": query, "fields": "title,url", "limit": limit}
    
    response = requests.get(url, params=params, timeout=10)
    
    if response.status_code != 200:
        st.error("Semantic Scholar API request failed.")
        return []

    papers = response.json().get("data", [])
    return [(paper.get("title", "N/A"), paper.get("url", "N/A")) for paper in papers]

def search_arxiv_papers(query, limit=5):
    """
    Fetches research papers from ArXiv.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    try:
        response = requests.get(BASE_URL_ARXIV, params=params, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        results = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            url = entry.find("{http://www.w3.org/2005/Atom}id").text
            results.append((title, url))
        
        return results
    except requests.exceptions.RequestException:
        st.error("ArXiv API request failed.")
        return []


# 🔐 MongoDB Connection
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["hypothesis_management"]
hypothesis_collection = db["hypotheses"]

# API Base URLs
BASE_URL_SEMANTIC = "https://api.semanticscholar.org/graph/v1"
BASE_URL_ARXIV = "http://export.arxiv.org/api/query"
BASE_URL_GOOGLE = "https://serpapi.com/search.json"

# 📌 UI - Evidence Recommender
st.title("Evidence Recommender")

# Step 1: Enter Hypothesis ID
with st.expander("Enter Hypothesis ID", expanded=True):
    hypothesis_id = st.text_input("Hypothesis ID:")
    if st.button("Load Hypothesis"):
        if not hypothesis_id.strip():
            st.warning("Please enter a valid Hypothesis ID.")
        else:
            st.session_state["loaded_hypothesis_id"] = hypothesis_id

# Get stored Hypothesis ID
loaded_hypothesis_id = st.session_state.get("loaded_hypothesis_id", None)

if loaded_hypothesis_id:
    # Fetch the hypothesis text from MongoDB
    hypothesis_entry = hypothesis_collection.find_one({"_id": loaded_hypothesis_id})
    if hypothesis_entry:
        hypothesis_text = hypothesis_entry["text"]
        st.success(f"**Loaded Hypothesis:** {hypothesis_text}")

        # Step 2: Select Sources
        with st.expander("Select Sources for Evidence Search", expanded=False):
            google_enabled = st.checkbox("Google Search")
            semantic_enabled = st.checkbox("Semantic Scholar")
            arxiv_enabled = st.checkbox("ArXiv")

        # API Key Input (Only shown if Google is selected)
        if google_enabled:
            google_api_key = st.text_input("Google API Key (SerpAPI)", type="password")
        else:
            google_api_key = None

        # 🚀 Fetch Evidence
        if st.button("Find Evidence"):
            with st.spinner("Searching for relevant evidence..."):
                recommended_evidence = []

                # ✅ Google Search
                if google_enabled and google_api_key:
                    st.write("Searching Google...")
                    google_results = search_google(google_api_key, hypothesis_text, top_n=5)
                    recommended_evidence.extend(google_results)

                # ✅ Semantic Scholar
                if semantic_enabled:
                    st.write("Searching Semantic Scholar...")
                    semantic_results = search_semantic_papers(hypothesis_text, limit=5)
                    recommended_evidence.extend(semantic_results)

                # ✅ ArXiv
                if arxiv_enabled:
                    st.write("Searching ArXiv...")
                    arxiv_results = search_arxiv_papers(hypothesis_text, limit=5)
                    recommended_evidence.extend(arxiv_results)

                # ✅ Display Results
                if recommended_evidence:
                    st.subheader("Recommended Evidence Files")
                    for title, url in recommended_evidence:
                        st.markdown(f"**[{title}]({url})**")
                else:
                    st.warning("No relevant evidence found. Try different sources or refine your hypothesis.")

    else:
        st.error("No hypothesis found for the given ID.")
