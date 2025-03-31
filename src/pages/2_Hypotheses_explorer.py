import streamlit as st
from pymongo import MongoClient
import gridfs
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId

# MongoDB Connection
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["infact_db_v3"]
fs = gridfs.GridFS(db)  # For file storage
hypothesis_collection = db["hypotheses"]

# Helper function to ensure ObjectId conversion
def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except Exception:
            pass  # Fall through to the return below
    # Return the original value if conversion failed or wasn't needed
    return id_value

# Add Claude-inspired CSS - clean, minimal, and elegant
st.markdown("""
<style>
    /* Set global text properties */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #2D3748;
        background-color: #F7FAFC;
    }
    
    /* Header styling */
    .main-header {
        color: #FF5733;
        font-size: 1.75rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #EDF2F7;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #4A5568 !important;
        background-color: #F7FAFC !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stExpander {
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Hypothesis styling */
    .hyp-id {
        font-size: 0.85rem;
        color: #718096;
        margin-bottom: 0.5rem;
    }
    
    .hyp-text {
        font-size: 1rem;
        margin-bottom: 1rem;
        color: #2D3748;
        line-height: 1.5;
    }
    
    .hyp-meta {
        font-size: 0.9rem;
        color: #718096;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #EDF2F7;
    }
    
    /* File container styling */
    .file-container {
        background-color: #F8F9FA;
        border-radius: 5px;
        padding: 0.75rem;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .file-item {
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0.5rem;
        border-radius: 4px;
        background-color: #FFFFFF;
        border-left: 3px solid #CBD5E0;
    }
    
    .file-name {
        color: #4A5568;
        font-weight: 500;
    }
    
    .status {
        color: #718096;
        font-size: 0.85rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1rem;
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
        color: #4A5568;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #F7FAFC;
        border: 1px solid #E2E8F0;
        color: #4A5568;
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #EDF2F7;
        border-color: #CBD5E0;
    }
    
    /* Info message styling */
    .stAlert {
        border-radius: 4px;
        padding: 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Hypothesis Explorer Page
st.markdown('<div class="main-header">Hypothesis Explorer</div>', unsafe_allow_html=True)

# Fetch all stored hypotheses
hypotheses = list(hypothesis_collection.find({}))

if hypotheses:
    st.write(f"Total Hypotheses: {len(hypotheses)}")

    for hypothesis in hypotheses:
        hypothesis_id = hypothesis["_id"]
        hypothesis_text = hypothesis["text"]
        
        # Count files specifically excluding node_state and rendered analysis files
        file_query = {
            "metadata.hypothesis_id": str(hypothesis_id),
            "filename": {"$not": {"$regex": "node_state|rendered_hypothesis"}}
        }
        
        # Get all files for hypothesis
        all_files = list(db.fs.files.find(file_query))
        
        # Collapsible Hypothesis Section
        with st.expander(f"Hypothesis {str(hypothesis_id)[-6:]}", expanded=False):
            # Display Hypothesis ID, Text and File Count with Claude-inspired styling
            st.markdown(f'<div class="hyp-id">ID: {hypothesis_id}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="hyp-text">{hypothesis_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="hyp-meta">Files Attached: {len(all_files)}</div>', unsafe_allow_html=True)
            
            if all_files:
                # Create a scrollable container for files
                with st.container(height=180, border=False):
                    st.markdown('<div class="file-container">', unsafe_allow_html=True)
                    for file in all_files:
                        filename = file.get("filename", "unnamed")
                        status = file.get("status", "unknown")
                        
                        # Improved file display
                        st.markdown(
                            f'<div class="file-item">'
                            f'<span class="file-name">{filename}</span>'
                            f'<span class="status">{status}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No files attached to this hypothesis.")
            
            # Find latest node state and rendered HTML
            latest_node_state = db.fs.files.find_one({
                "metadata.type": "node_state",
                "metadata.hypothesis_id": str(hypothesis_id),
                "metadata.is_latest": True
            })
            
            latest_html = db.fs.files.find_one({
                "metadata.type": "rendered_html",
                "metadata.hypothesis_id": str(hypothesis_id),
                "metadata.is_latest": True
            })
            
            # Download buttons with consistent styling
            st.markdown('<div class="section-header">Analysis Downloads</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                if latest_node_state:
                    # Get the file data
                    node_state_data = fs.get(ensure_object_id(latest_node_state["_id"])).read()
                    st.download_button(
                        label="Download Node State (JSON)",
                        data=node_state_data,
                        file_name=f"hypothesis_{hypothesis_id}_state.json",
                        mime="application/json",
                        key=f"node_state_{hypothesis_id}",
                        use_container_width=True
                    )
                else:
                    st.info("No node state available")
                
            with col2:
                if latest_html:
                    # Get the file data
                    html_data = fs.get(ensure_object_id(latest_html["_id"])).read()
                    st.download_button(
                        label="Download Visualization (HTML)",
                        data=html_data,
                        file_name=f"hypothesis_{hypothesis_id}_visualization.html",
                        mime="text/html",
                        key=f"html_{hypothesis_id}",
                        use_container_width=True
                    )
                else:
                    st.info("No visualization available")

else:
    st.warning("No hypotheses found in the database.")