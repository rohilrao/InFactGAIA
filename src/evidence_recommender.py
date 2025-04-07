import requests
import time
import json
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple

class EvidenceRecommender:
    """
    A recommender system for scientific evidence based on a hypothesis.
    Uses multiple scientific search APIs to find relevant papers with numerical evidence.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the evidence recommender with API keys.
        
        Args:
            api_keys: Dictionary containing API keys for different services
                      (e.g., {"arxiv": "key", "semantic_scholar": "key"})
        """
        self.api_keys = api_keys or {}
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
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(base_url, params=params)
            
            if response.status_code != 200:
                st.error(f"ArXiv API error: {response.status_code}")
                return []
                
            # Parse XML response (arXiv API returns Atom XML)
            # This is a simplified parsing approach - production code should use proper XML parsing
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
                    'confidence_score': self._calculate_confidence_score(query, title, summary)
                }
                
                results.append(result)
                
            return results
            
        except Exception as e:
            st.error(f"Error searching arXiv: {str(e)}")
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
                if response.status_code == 429:
                    st.warning("Semantic Scholar API rate limit exceeded. Try again later.")
                else:
                    st.error(f"Semantic Scholar API error: {response.status_code}")
                return []
                
            data = response.json()
            
            results = []
            for paper in data.get('data', []):
                authors = [author.get('name', '') for author in paper.get('authors', [])]
                
                # Create result dictionary
                result = {
                    'title': paper.get('title', 'No title'),
                    'summary': paper.get('abstract', 'No abstract available')[:300] + '...' if paper.get('abstract', '') and len(paper.get('abstract', '')) > 300 else paper.get('abstract', 'No abstract available'),
                    'authors': authors[:3],  # Limit to first 3 authors
                    'published_date': paper.get('publicationDate', paper.get('year', 'Unknown')),
                    'pdf_url': paper.get('url', ''),
                    'doi': paper.get('externalIds', {}).get('DOI', ''),
                    'source': 'semantic_scholar',
                    'category': paper.get('venue', 'Unknown'),
                    'is_open_access': paper.get('isOpenAccess', False),
                    'confidence_score': self._calculate_confidence_score(query, paper.get('title', ''), paper.get('abstract', ''))
                }
                
                results.append(result)
                
            return results
            
        except Exception as e:
            st.error(f"Error searching Semantic Scholar: {str(e)}")
            return []
    
    def _calculate_confidence_score(self, query: str, title: str, abstract: str) -> float:
        """
        Calculate a simple confidence score for how well the paper matches the query.
        
        Args:
            query: The search query
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            Confidence score between 0 and 1
        """
        # This is a simplified scoring method - in production, you'd want more sophisticated NLP
        
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
                               'correlation', 'regression', 'mean', 'median', 'standard deviation']
        
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
            sources = ['arxiv', 'semantic_scholar']
            
        all_results = []
        
        # Generate search query from hypothesis
        search_query = self._generate_search_query(hypothesis)
        
        # Search each source
        if 'arxiv' in sources:
            arxiv_results = self.search_arxiv(search_query, self.max_results)
            all_results.extend(arxiv_results)
            
        if 'semantic_scholar' in sources:
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            semantic_results = self.search_semantic_scholar(search_query, self.max_results)
            all_results.extend(semantic_results)
            
        # Sort by confidence score (descending)
        sorted_results = sorted(all_results, key=lambda x: x.get('confidence_score', 0), reverse=True)
        
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


def display_evidence_recommendations(hypothesis_text: str, api_keys: Dict[str, str] = None) -> Tuple[bool, Optional[Dict]]:
    """
    Display evidence recommendations in the Streamlit UI and return selected evidence
    
    Args:
        hypothesis_text: The hypothesis text
        api_keys: Dictionary of API keys for different services
        
    Returns:
        Tuple containing:
            - Boolean indicating if evidence was selected
            - Selected evidence metadata (if any)
    """
    st.markdown("### :orange[Evidence Recommendations]")
    
    evidence_selected = False
    selected_evidence = None
    
    # Initialize evidence recommender
    recommender = EvidenceRecommender(api_keys)
    
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
                    st.warning("No relevant papers found. Try rephrasing your hypothesis.")
        
        # Display recommendations if available in session state
        if 'evidence_recommendations' in st.session_state:
            recommendations = st.session_state['evidence_recommendations']
            
            if recommendations:
                st.success(f"Found {len(recommendations)} relevant papers")
                
                # Create a container with scrollable area
                papers_container = st.container(height=400, border=True)
                
                with papers_container:
                    # Display each recommendation
                    for i, paper in enumerate(recommendations):
                        col1, col2 = st.columns([5, 1])
                        
                        with col1:
                            # Paper title and basic info
                            st.markdown(f"**{paper['title']}**")
                            st.caption(f"Authors: {', '.join(paper['authors'])} | Published: {paper['published_date']} | Source: {paper['source'].title()}")
                            
                            # Confidence score display
                            confidence = paper['confidence_score']
                            score_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                            st.markdown(f"Relevance: <span style='color:{score_color}'>{confidence:.2f}</span>", unsafe_allow_html=True)
                            
                            # Summary
                            st.markdown(f"*{paper['summary']}*")
                            
                            # Links
                            if paper.get('pdf_url'):
                                st.markdown(f"[Download PDF]({paper['pdf_url']})")
                            
                            if paper.get('doi'):
                                st.markdown(f"[DOI Link]({paper['doi']})")
                        
                        with col2:
                            # Selection button
                            if st.button("Use", key=f"use_paper_{i}", use_container_width=True):
                                evidence_selected = True
                                selected_evidence = paper
                                st.session_state['selected_evidence'] = paper
                                st.success(f"Selected paper: {paper['title']}")
                                
                        # Divider between papers
                        if i < len(recommendations) - 1:
                            st.divider()
            else:
                st.info("No papers found. Try adjusting your hypothesis or search terms.")
        
        # Display selected evidence
        if 'selected_evidence' in st.session_state:
            with st.container(border=True):
                paper = st.session_state['selected_evidence']
                st.markdown("#### Selected Evidence")
                st.markdown(f"**{paper['title']}**")
                st.caption(f"Authors: {', '.join(paper['authors'])} | Published: {paper['published_date']}")
                st.markdown(f"*{paper['summary']}*")
                
                if paper.get('pdf_url'):
                    st.markdown(f"[Download PDF]({paper['pdf_url']})")
                
                if st.button("Clear Selection"):
                    if 'selected_evidence' in st.session_state:
                        del st.session_state['selected_evidence']
                    evidence_selected = False
                    selected_evidence = None
                    st.rerun()
    
    return evidence_selected, selected_evidence

