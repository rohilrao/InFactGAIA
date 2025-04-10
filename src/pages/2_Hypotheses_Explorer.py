import streamlit as st
from pymongo import MongoClient
import gridfs
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
import pandas as pd
import json
from datetime import datetime

# -------------------
# Streamlit Page Config
# -------------------
st.set_page_config(
    page_title="GAIA - InFact - Demo",
    page_icon="",
    layout="wide"
)

# -------------------
# MongoDB Connection
# -------------------
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["infact_db_v3"]
fs = gridfs.GridFS(db)
hypothesis_collection = db["hypotheses"]

def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if valid."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except Exception:
            pass
    return id_value

# -------------------
# Page Header (Orange via Streamlit Markdown)
# -------------------
st.markdown("### :orange[Hypothesis Explorer]")

# -------------------
# Hypothesis Explorer
# -------------------
hypotheses = list(hypothesis_collection.find({}))

if hypotheses:
    st.write(f"**Total Hypotheses:** {len(hypotheses)}")

    for hypothesis in hypotheses:
        hypothesis_id = hypothesis["_id"]
        hypothesis_text = hypothesis["text"]
        
        # Query for files related to this hypothesis
        # Look for the hypothesis_id in the metadata field
        file_query = {
            "metadata.hypothesis_id": str(hypothesis_id),
            "filename": {"$not": {"$regex": "node_state|rendered_hypothesis"}}
        }
        all_files = list(db.fs.files.find(file_query))
        
        with st.expander(f"Hypothesis ID: :orange[{hypothesis_id}]", expanded=False):
            # Highlight the hypothesis text in orange
            st.markdown(f"**Hypothesis:** :orange[{hypothesis_text}]")
            
            # Update file query to exclude visualization.html files
            file_query = {
            "metadata.hypothesis_id": str(hypothesis_id),
            "filename": {
                "$not": {"$regex": "node_state|rendered_hypothesis|_visualization\\.html$"}
            }
            }
            all_files = list(db.fs.files.find(file_query))
            
            # Simple count of files
            st.markdown(f"**Files Attached:** {len(all_files)}")
            if all_files:
                # Create scrollable container for files
                file_container = st.container(height=200, border=True)
                
                # Custom CSS to reduce spacing between items
                st.markdown("""
                <style>
                    .compact-items p {
                    margin-bottom: 0px;
                    padding-bottom: 0px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                with file_container:
                    # Add compact-items class to reduce vertical spacing
                    st.markdown('<div class="compact-items">', unsafe_allow_html=True)
                    for file in all_files:
                        filename = file.get("filename", "unnamed")
                        
                        # Properly access the status field (could be at top level or in metadata)
                        status = file.get("status", "unknown")
                        if status == "unknown" and "metadata" in file and "status" in file["metadata"]:
                            status = file["metadata"]["status"]
                        
                        col1, col2 = st.columns([3, 2])
                        with col1:
                            st.markdown(f"**{filename}**")
                        with col2:
                            # Always show status in green as requested
                            st.markdown(f":green[{status}]")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No files attached to this hypothesis.")
                
            # Latest node state & rendered HTML
            latest_node_state = db.fs.files.find_one({
                "metadata.type": "node_state",
                "metadata.hypothesis_id": str(hypothesis_id),
                "metadata.is_latest": True
            })
            
            # If we don't find a node state file, we can use the hypothesis data itself
            latest_html = db.fs.files.find_one({
                "metadata.type": "rendered_html",
                "metadata.hypothesis_id": str(hypothesis_id)
            }, sort=[("metadata.rendered_at", -1)]) # Get the latest HTML file based on rendered_at timestamp
            
            # Download Buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if latest_node_state:
                    # Download the node state file if it exists
                    node_state_data = fs.get(ensure_object_id(latest_node_state["_id"])).read()
                    st.download_button(
                        label="Download Node State (JSON)",
                        data=node_state_data,
                        file_name=f"hypothesis_{hypothesis_id}_state.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    # If no node state file exists, create a JSON from the hypothesis document
                    # Create a serializable version of the hypothesis
                    def serialize_for_json(obj):
                        if isinstance(obj, ObjectId):
                            return str(obj)
                        elif isinstance(obj, datetime):
                            return obj.isoformat()
                        elif isinstance(obj, dict):
                            return {k: serialize_for_json(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [serialize_for_json(item) for item in obj]
                        elif isinstance(obj, (str, int, float, bool, type(None))):
                            return obj
                        else:
                            return str(obj)
                    
                    serializable_hypothesis = serialize_for_json(hypothesis)
                    hypothesis_json = json.dumps(serializable_hypothesis, indent=2)
                    
                    st.download_button(
                        label="Download InFact Node State (JSON)",
                        data=hypothesis_json,
                        file_name=f"hypothesis_{hypothesis_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            with col2:
                # Get rendered HTML directly from the hypothesis document
                latest_html_content = hypothesis.get("node_metadata", {}).get("latest_rendered_html")
                latest_rendered_at = hypothesis.get("node_metadata", {}).get("rendered_at")
                
                if latest_html_content:
                    # Use the HTML content stored directly in the document
                    st.download_button(
                        label="Download Visualization (HTML)",
                        data=latest_html_content,
                        file_name=f"hypothesis_{hypothesis_id}_visualization.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    # Show when it was last rendered if timestamp is available
                    if latest_rendered_at:
                        st.caption(f"Last rendered: {latest_rendered_at.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.info("No visualization available")

else:
    st.warning("No hypotheses found in the database.")