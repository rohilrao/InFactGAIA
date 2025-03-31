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

# Add minimal CSS - consistent font sizes and subtle colors
st.markdown("""
<style>
    /* Set consistent text properties */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        font-weight: normal !important;
    }
    .stExpander {
        border: none !important;
    }
    .hyp-text {
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    .file-container {
        padding: 0.5rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .file-item {
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
        display: flex;
        justify-content: space-between;
    }
    .file-name {
        color: #d6d6d6;
    }
    .status {
        color: #a0a0a0;
    }
    .section-header {
        font-size: 0.95rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #d6d6d6;
    }
</style>
""", unsafe_allow_html=True)

# Hypothesis Explorer Page
st.markdown("### Hypothesis Explorer")

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
        with st.expander(f"Hypothesis ID: {hypothesis_id}", expanded=False):
            # Display Hypothesis Text and File Count with consistent styling
            st.markdown(f'<div class="hyp-text">Hypothesis: {hypothesis_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="hyp-text">Files Attached: {len(all_files)}</div>', unsafe_allow_html=True)
            
            if all_files:
                # Create a scrollable container for files
                with st.container(height=150, border=False):
                    st.markdown('<div class="file-container">', unsafe_allow_html=True)
                    for file in all_files:
                        filename = file.get("filename", "unnamed")
                        status = file.get("status", "unknown")
                        
                        # Simplified file display
                        st.markdown(
                            f'<div class="file-item">'
                            f'<span class="file-name">{filename}</span>'
                            f'<span class="status">{status}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.write("No files attached to this hypothesis.")
            
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