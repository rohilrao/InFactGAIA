import requests
import time
import json
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import os

# Import the call_llm function from infact_utils
from infact_utils import call_llm

class EvidenceRecommender:
    """
    A recommender system for scientific evidence based on a hypothesis.
    Uses multiple scientific search APIs to find relevant papers with numerical evidence.
    """
    
    def __init__(self, api_key: str = None, provider: str = "openai", model: str = "gpt-4o", api_keys: Dict[str, str] = None):
        """
        Initialize the evidence recommender with API keys.
        
        Args:
            api_key: The main API key for LLM calls
            provider: The LLM provider (openai, anthropic, etc.)
            model: The LLM model to use
            api_keys: Additional API keys for different services (serpapi, semantic_scholar)
        """
        # Store LLM settings
        self.provider = provider
        self.model = model
        
        # Initialize api_keys dictionary
        self.api_keys = api_keys or {}
        
        # Add main API key to appropriate provider
        if api_key:
            self.api_keys[provider.lower()] = api_key
        
        # Try to load SerpAPI key from secrets if not provided
        if 'serpapi' not in self.api_keys and hasattr(st, 'secrets') and 'SERPAPI_KEY' in st.secrets:
            self.api_keys['serpapi'] = st.secrets['SERPAPI_KEY']
            
        # Set default result counts
        self.max_results = 5
        
    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers related to the hypothesis.
        
        Args:
            query: The search query (based on hypothesis)
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        try:
            # arXiv API doesn't require authentication
            base_url = "http://export.arxiv.org/api/query"
            
            # Add terms to focus on papers with numerical evidence
            enhanced_query = f"{query} AND (data OR statistics OR measurement OR numerical OR quantitative OR analysis)"
            
            params = {
                "search_query": enhanced_query,
                "start": 0,
                "max_results": max_results * 2,  # Request more to filter by confidence score
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(base_url, params=params)
            
            if response.status_code != 200:
                # Silently handle errors
                return []
                
            # Parse XML response (arXiv API returns Atom XML)
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(response.content)
            
            # Define namespace for arXiv Atom feed
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            results = []
            for entry in root.findall('.//atom:entry', ns):
                title = entry.find('./atom:title', ns).text.strip()
                summary = entry.find('./atom:summary', ns).text.strip()
                published = entry.find('./atom:published', ns).text.strip()
                
                # Calculate confidence score - updated to use LLM if available
                confidence_score = self._calculate_confidence_score(query, title, summary)
                
                # Only include if score is high enough
                if confidence_score < 0.75:
                    continue
                
                # Get PDF link
                links = entry.findall('./atom:link', ns)
                pdf_url = ""
                for link in links:
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href')
                        break
                
                # Get DOI if available
                doi = ""
                for link in links:
                    if link.get('rel') == 'related' and 'doi' in link.get('href', ''):
                        doi = link.get('href')
                        break
                
                # Get authors
                authors = []
                for author in entry.findall('./atom:author/atom:name', ns):
                    authors.append(author.text.strip())
                
                # Get primary category
                primary_category = entry.find('./arxiv:primary_category', ns).get('term')
                
                # Create result dictionary
                result = {
                    'title': title,
                    'summary': summary[:300] + '...' if len(summary) > 300 else summary,
                    'authors': authors[:3],  # Limit to first 3 authors
                    'published_date': published.split('T')[0],  # Just get the date part
                    'pdf_url': pdf_url,
                    'doi': doi,
                    'source': 'arxiv',
                    'category': primary_category,
                    'confidence_score': confidence_score
                }
                
                results.append(result)
                
            return results
            
        except Exception as e:
            # Silently handle exceptions
            return []
    
    def search_semantic_scholar(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for papers related to the hypothesis.
        
        Args:
            query: The search query (based on hypothesis)
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        try:
            # Simplified approach - in production, you should use the official Semantic Scholar API client
            api_key = self.api_keys.get('semantic_scholar')
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            
            # Enhanced query to find papers with numerical evidence
            enhanced_query = f"{query} (data OR statistics OR quantitative OR numerical)"
            
            headers = {}
            if api_key:
                headers['x-api-key'] = api_key
                
            params = {
                'query': enhanced_query,
                'limit': max_results,
                'fields': 'title,abstract,authors,url,year,venue,isOpenAccess,publicationDate',
            }
            
            response = requests.get(base_url, headers=headers, params=params)
            
            if response.status_code != 200:
                # Silently handle errors without showing warnings
                return []
                
            data = response.json()
            
            results = []
            for paper in data.get('data', []):
                authors = [author.get('name', '') for author in paper.get('authors', [])]
                
                title = paper.get('title', 'No title')
                abstract = paper.get('abstract', 'No abstract available')
                
                # Calculate confidence score - updated to use LLM if available
                confidence_score = self._calculate_confidence_score(query, title, abstract)
                
                # Only include if score is high enough
                if confidence_score < 0.75:
                    continue
                
                # Create result dictionary
                result = {
                    'title': title,
                    'summary': abstract[:300] + '...' if abstract and len(abstract) > 300 else abstract,
                    'authors': authors[:3],  # Limit to first 3 authors
                    'published_date': paper.get('publicationDate', paper.get('year', 'Unknown')),
                    'pdf_url': paper.get('url', ''),
                    'doi': paper.get('externalIds', {}).get('DOI', ''),
                    'source': 'semantic_scholar',
                    'category': paper.get('venue', 'Unknown'),
                    'is_open_access': paper.get('isOpenAccess', False),
                    'confidence_score': confidence_score
                }
                
                results.append(result)
                
            return results
            
        except Exception as e:
            # Silently handle exceptions without showing errors
            return []
    
    def search_serpapi(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Google using SerpAPI for papers related to the hypothesis.
        
        Args:
            query: The search query (based on hypothesis)
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        try:
            api_key = self.api_keys.get('serpapi')
            if not api_key:
                return []
                
            base_url = "https://serpapi.com/search"
            
            # Enhanced query to find papers with numerical evidence
            enhanced_query = f"{query} scientific paper data evidence research filetype:pdf"
            
            params = {
                "api_key": api_key,
                "q": enhanced_query,
                "num": max_results * 2,  # Request more to filter by relevance
                "engine": "google_scholar" if "thesis" in query or "research" in query else "google",
            }
            
            response = requests.get(base_url, params=params)
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            
            results = []
            organic_results = data.get("organic_results", [])
            
            for result in organic_results:
                title = result.get("title", "No title")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                
                # Only process PDF links or likely academic sources
                is_pdf = link.endswith(".pdf")
                is_academic = any(domain in link for domain in [
                    "scholar.google", "researchgate", "academia.edu", "arxiv.org",
                    "nature.com", "science.org", "sciencedirect.com", "springer.com",
                    "ncbi.nlm.nih.gov", "pubmed", "ieee.org"
                ])
                
                if not (is_pdf or is_academic):
                    continue
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(query, title, snippet)
                
                # Only include if score is high enough
                if confidence_score < 0.75:
                    continue
                
                # Create result dictionary with available information
                result_dict = {
                    'title': title,
                    'summary': snippet[:300] + '...' if len(snippet) > 300 else snippet,
                    'authors': [],  # SerpAPI doesn't consistently provide authors
                    'published_date': 'Unknown',  # SerpAPI doesn't consistently provide dates
                    'pdf_url': link if is_pdf else "",
                    'doi': "",
                    'source': 'google_search',
                    'category': 'Unknown',
                    'confidence_score': confidence_score
                }
                
                results.append(result_dict)
                
                # Only take the requested number of results
                if len(results) >= max_results:
                    break
                    
            return results
            
        except Exception as e:
            # Silently handle exceptions
            return []
    
    def _calculate_confidence_score(self, query: str, title: str, abstract: str) -> float:
        """
        Calculate a confidence score for how well the paper matches the query.
        Uses LLM when available or falls back to heuristic matching.
        
        Args:
            query: The search query
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            Confidence score between 0 and 1
        """
        # Try to use LLM for more sophisticated matching if we have an API key for current provider
        api_key = self.api_keys.get(self.provider.lower())
        
        if api_key and abstract:
            try:
                # Create prompt for LLM to evaluate relevance
                prompt = f"""
                Evaluate the relevance and quality of this scientific paper for the given hypothesis.
                
                HYPOTHESIS: "{query}"
                
                PAPER TITLE: "{title}"
                
                ABSTRACT: "{abstract}"
                
                Please judge:
                1. How relevant this paper is to the hypothesis (considering if it directly addresses the hypothesis)
                2. If the paper contains numerical or statistical evidence to support or refute the hypothesis
                3. The overall quality and reliability of the evidence
                
                Return a single number between 0.0 and 1.0 representing the confidence score, where:
                - 0.0-0.5: Not relevant or poor quality evidence
                - 0.5-0.7: Somewhat relevant but limited numerical evidence
                - 0.7-0.85: Relevant with good numerical evidence
                - 0.85-1.0: Highly relevant with strong numerical evidence
                
                Output only the numerical score without any explanation.
                """
                
                # Call LLM with the current provider and model settings
                result = call_llm(self.provider, api_key, self.model, prompt)
                
                # Extract score - handle various formats the LLM might return
                try:
                    # Clean the result string and extract the numerical value
                    cleaned_result = result.strip().replace('Score:', '').replace('Confidence Score:', '').strip()
                    score = float(cleaned_result)
                    return max(0.0, min(score, 1.0))  # Ensure value is between 0 and 1
                except:
                    # If parsing fails, fall back to heuristic method
                    pass
            except:
                # Silently fall back to heuristic method if LLM call fails
                pass
        
        # Fallback: Heuristic method for calculating score
        
        # Convert to lowercase for comparison
        query_terms = query.lower().split()
        title_lower = title.lower()
        abstract_lower = abstract.lower() if abstract else ""
        
        # Count query terms in title (weighted higher)
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = title_matches / len(query_terms) if query_terms else 0
        
        # Count query terms in abstract
        abstract_matches = sum(1 for term in query_terms if term in abstract_lower)
        abstract_score = abstract_matches / len(query_terms) if query_terms else 0
        
        # Check for numerical content
        numerical_indicators = ['data', 'statistics', 'analysis', 'measurement', 'experiment', 
                               'study', 'evidence', 'survey', 'sample', 'p-value', 'significant', 
                               'correlation', 'regression', 'mean', 'median', 'standard deviation',
                               '%', 'percent', 'percentage', 'rate', 'ratio', 'average', 'trend']
        
        numerical_score = 0
        for indicator in numerical_indicators:
            if indicator in abstract_lower:
                numerical_score += 0.05  # Add a small score for each indicator
        
        numerical_score = min(numerical_score, 0.5)  # Cap at 0.5
        
        # Combined score (weighted: title 40%, abstract 30%, numerical 30%)
        combined_score = (title_score * 0.4) + (abstract_score * 0.3) + numerical_score
        
        # Return score capped at 1.0
        return min(combined_score, 1.0)
    
    def recommend_evidence(self, hypothesis: str, sources: List[str] = None) -> List[Dict[str, Any]]:
        """
        Recommend scientific papers as evidence for the given hypothesis.
        
        Args:
            hypothesis: The hypothesis text
            sources: List of sources to search (default: all available sources)
            
        Returns:
            List of recommended papers with metadata
        """
        if not sources:
            sources = ['arxiv', 'serpapi', 'semantic_scholar']
            
        all_results = []
        
        # Generate search query from hypothesis
        search_query = self._generate_search_query(hypothesis)
        
        # Search each source
        if 'arxiv' in sources:
            arxiv_results = self.search_arxiv(search_query, self.max_results)
            all_results.extend(arxiv_results)
            
        if 'serpapi' in sources and 'serpapi' in self.api_keys:
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            serpapi_results = self.search_serpapi(search_query, self.max_results)
            all_results.extend(serpapi_results)
            
        if 'semantic_scholar' in sources:
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            semantic_results = self.search_semantic_scholar(search_query, self.max_results)
            all_results.extend(semantic_results)
            
        # Filter for high confidence and sort by confidence score (descending)
        high_confidence_results = [r for r in all_results if r.get('confidence_score', 0) >= 0.75]
        sorted_results = sorted(high_confidence_results, key=lambda x: x.get('confidence_score', 0), reverse=True)
        
        # Return top results
        return sorted_results[:self.max_results]
    
    def _generate_search_query(self, hypothesis: str) -> str:
        """
        Generate a search query from the hypothesis.
        
        Args:
            hypothesis: The hypothesis text
            
        Returns:
            A search query optimized for scientific paper search
        """
        # Remove common question words and phrases
        question_words = ['is', 'are', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 
                         'the', 'a', 'an', 'that', 'this', 'these', 'those', 'to', 'of']
        
        # Convert to lowercase and tokenize
        tokens = hypothesis.lower().split()
        
        # Remove question words
        filtered_tokens = [token for token in tokens if token not in question_words]
        
        # Join tokens back into a query
        query = ' '.join(filtered_tokens)
        
        return query


def display_evidence_recommendations(hypothesis_text: str, api_key: str = None, provider: str = "openai", model: str = "gpt-4o", api_keys: Dict[str, str] = None) -> Tuple[bool, Optional[Dict]]:
    """
    Display evidence recommendations in the Streamlit UI and return selected evidence
    
    Args:
        hypothesis_text: The hypothesis text
        api_key: Main API key for the current provider
        provider: LLM provider name (openai, anthropic, etc.)
        model: Model name to use with the provider
        api_keys: Additional API keys for different services
        
    Returns:
        Tuple containing:
            - Boolean indicating if evidence was selected
            - Selected evidence metadata (if any)
    """
    st.markdown("### :orange[Evidence Recommendations]")
    
    evidence_selected = False
    selected_evidence = None
    
    # Ensure we have an api_keys dictionary
    if api_keys is None:
        api_keys = {}
        
    # Add the main API key to the appropriate provider
    if api_key and provider:
        api_keys[provider.lower()] = api_key
    
    # Initialize evidence recommender with provider settings
    recommender = EvidenceRecommender(api_key=api_key, provider=provider, model=model, api_keys=api_keys)
    
    # Create expander UI element to keep the UI compact
    with st.expander("🔍 Find supporting scientific evidence for your hypothesis", expanded=True):
        st.write(f"Finding papers related to: **{hypothesis_text}**")
        
        # Add option to adjust number of results
        col1, col2 = st.columns([3, 1])
        with col2:
            max_results = st.slider("Max results", min_value=3, max_value=10, value=5, step=1)
            recommender.max_results = max_results
        
        # Search button
        if st.button("Search for Evidence", use_container_width=True):
            with st.spinner("Searching for scientific papers..."):
                # Get recommendations
                recommendations = recommender.recommend_evidence(hypothesis_text)
                
                # Store in session state
                st.session_state['evidence_recommendations'] = recommendations
                
                if not recommendations:
                    st.info("No high-confidence evidence found. Try rephrasing your hypothesis or checking more general scientific sources.")
        
        # Display recommendations if available in session state
        if 'evidence_recommendations' in st.session_state:
            recommendations = st.session_state['evidence_recommendations']
            
            if recommendations:
                st.success(f"Found {len(recommendations)} relevant papers with high confidence scores (≥0.75)")
                
                # Create a container with scrollable area
                papers_container = st.container(height=400, border=True)
                
                with papers_container:
                    # Display each recommendation
                    for i, paper in enumerate(recommendations):
                        # Paper title and basic info
                        st.markdown(f"**{paper['title']}**")
                        
                        # Show authors if available
                        if paper.get('authors') and len(paper['authors']) > 0:
                            authors_text = f"Authors: {', '.join(paper['authors'])}"
                        else:
                            authors_text = ""
                            
                        # Show source info
                        source_text = f"Source: {paper['source'].replace('_', ' ').title()}"
                        
                        # Combine metadata
                        meta_parts = [p for p in [authors_text, f"Published: {paper['published_date']}", source_text] if p]
                        st.caption(" | ".join(meta_parts))
                        
                        # Confidence score display
                        confidence = paper['confidence_score']
                        score_color = "green" if confidence >= 0.85 else "orange" if confidence >= 0.75 else "red"
                        st.markdown(f"Relevance: <span style='color:{score_color}'>{confidence:.2f}</span>", unsafe_allow_html=True)
                        
                        # Summary
                        st.markdown(f"*{paper['summary']}*")
                        
                        # Links row
                        links = []
                        if paper.get('pdf_url'):
                            links.append(f"[Download PDF]({paper['pdf_url']})")
                        
                        if paper.get('doi'):
                            links.append(f"[DOI Link]({paper['doi']})")
                            
                        if links:
                            st.markdown(" | ".join(links))
                        
                        # Divider between papers
                        if i < len(recommendations) - 1:
                            st.divider()
            else:
                st.info("No high-confidence evidence found. Try adjusting your hypothesis or search terms.")
        
        # Instead of storing selected evidence, automatically make the first result available
        if 'evidence_recommendations' in st.session_state and st.session_state['evidence_recommendations']:
            # Use the highest ranked paper as the selected evidence
            selected_evidence = st.session_state['evidence_recommendations'][0]
            evidence_selected = True
            
            # Store in session state
            st.session_state['selected_evidence'] = selected_evidence
    
    return evidence_selected, selected_evidence
