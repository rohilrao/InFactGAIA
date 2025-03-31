import requests
import time
import random
import xml.etree.ElementTree as ET
import os
from urllib.parse import urlparse

def search_google(api_key, query, top_n=10):
    """
    Fetches search results from SerpAPI based on the provided query and API key.
    
    Parameters:
    - api_key (str): Your SerpAPI key.
    - query (str): The search query.
    - top_n (int): Number of top URLs to print.
    
    Returns:
    - None: Prints the top_n URLs.
    """
    url = f"https://serpapi.com/search.json?engine=google&q={query}&api_key={api_key}"
    
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching data:", response.status_code)
        return
    
    data = response.json()
    
    # Extract URLs
    urls = [result["link"] for result in data.get("organic_results", [])]
    
    # Print top N URLs
    for url in urls[:top_n]:
        print(url)


BASE_URL_SEMANTIC = "https://api.semanticscholar.org/graph/v1"

def search_semantic_papers(query, limit=10, retries=5, backoff_factor=2, max_wait=120):
    """Fetch papers from Semantic Scholar API and print title + URL."""
    url = f"{BASE_URL_SEMANTIC}/paper/search"
    params = {"query": query, "fields": "title,url", "limit": limit}

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get("Retry-After", backoff_factor * (2 ** attempt)))
                wait_time = min(retry_after + random.uniform(0, 2), max_wait)
                print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                continue  # Retry the request

            response.raise_for_status()  # Raise exception for non-200 responses
            
            papers = response.json().get("data", [])

            for paper in papers:
                print(paper.get("title", "N/A"), "-", paper.get("url", "N/A"))

            return
        except requests.exceptions.RequestException as e:
            wait_time = min(backoff_factor * (2 ** attempt) + random.uniform(0, 2), max_wait)
            print(f"Request failed ({e}). Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

    print("Max retries reached. Exiting.")

BASE_URL_ARXIV = "http://export.arxiv.org/api/query"

def search_arxiv_papers(query, limit=100):
    """Fetch papers from ArXiv API and print title + URL."""
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

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            url = entry.find("{http://www.w3.org/2005/Atom}id").text
            print(title, "-", url)

    except requests.exceptions.RequestException as e:
        print(f"ArXiv API request failed: {e}")





def download_file(url, target_directory):
    """
    Downloads a file (PDF or HTML) from the given URL and saves it to the target directory.

    Args:
        url (str): The URL of the file (PDF, HTML, etc.).
        target_directory (str): The directory where the file will be saved.

    Returns:
        str: The full path of the saved file, or None if the download fails.
    """
    try:
        # Ensure target directory exists
        if not os.path.exists(target_directory):
            print(f"⚠️ Directory does not exist. Creating: {target_directory}")
            os.makedirs(target_directory)

        # Handle ArXiv abstract -> Convert to PDF link
        if "/abs/" in url:
            url = url.replace("/abs/", "/pdf/") + ".pdf"

        # Parse URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "download"

        # Ensure the file has the correct extension
        if filename.endswith(".pdf"):
            extension = ".pdf"
        elif filename.endswith(".html") or "semanticscholar" in url or "google" in url:
            extension = ".html"
        else:
            extension = ".html"  # Default to HTML if no clear extension

        # Add extension if missing
        if not filename.endswith(extension):
            filename += extension

        # Final file path
        filepath = os.path.join(target_directory, filename)

        # Send request
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        # Ensure content is not empty before saving
        content = response.content if extension == ".pdf" else response.text
        if not content.strip():  # If content is empty, skip saving
            print(f"❌ Download failed: {url} (Empty content)")
            return None

        # Save file (binary for PDFs, text for HTML)
        mode = "wb" if extension == ".pdf" else "w"
        encoding = None if extension == ".pdf" else "utf-8"

        with open(filepath, mode, encoding=encoding) as file:
            file.write(content)

        print(f"✅ Downloaded: {url} -> {filepath}")
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {url}: {e}")
        return None
