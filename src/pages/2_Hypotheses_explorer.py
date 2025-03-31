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

# Add some custom CSS
st.markdown("""
<style>
    .hyp-header {
        color: #FF8800;
        font-size: 1rem;
        font-weight: bold;
    }
    .file-name {
        color: #3498db;
        font-weight: 500;
    }
    .status-processed {
        color: #27ae60;
        font-weight: 500;
    }
    .status-ready {
        color: #f39c12;
        font-weight: 500;
    }
    .status-unprocessed {
        color: #7f8c8d;
        font-weight: 500;
    }
    .status-unknown {
        color: #95a5a6;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Hypothesis Explorer Page
st.markdown("### :orange[Hypothesis Explorer]")

# Fetch all stored hypotheses
hypotheses = list(hypothesis_collection.find({}))

if hypotheses:
    st.write(f"**Total Hypotheses:** {len(hypotheses)}")

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
            st.markdown(f'<span class="hyp-header">Hypothesis:</span> {hypothesis_text}', unsafe_allow_html=True)
            st.markdown(f'<span class="hyp-header">Files Attached:</span> {len(all_files)}', unsafe_allow_html=True)
            
            if all_files:
                # Create a scrollable container for files
                with st.container(height=200, border=True):
                    for file in all_files:
                        filename = file.get("filename", "unnamed")
                        status = file.get("status", "unknown")
                        
                        # Apply different styling based on status
                        status_class = "status-unknown"
                        if status == "processed":
                            status_class = "status-processed"
                        elif status == "ready_for_analysis":
                            status_class = "status-ready"
                        elif status == "unprocessed":
                            status_class = "status-unprocessed"
                        
                        # Display file with colored status (without "Status:" prefix)
                        st.markdown(
                            f'<span class="file-name">{filename}</span> — '
                            f'<span class="{status_class}">{status}</span>',
                            unsafe_allow_html=True
                        )
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
            st.markdown('<span class="hyp-header">Analysis Downloads:</span>', unsafe_allow_html=True)
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
                        key=f"node_state_{hypothesis_id}"
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
                        key=f"html_{hypothesis_id}"
                    )
                else:
                    st.info("No visualization available")

else:
    st.warning("No hypotheses found in the database.")