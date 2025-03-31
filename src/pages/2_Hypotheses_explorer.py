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
        
        # Count files by status
        processed_files = sum(1 for file in all_files if file.get("status") == "processed")
        unprocessed_files = sum(1 for file in all_files if file.get("status") == "unprocessed")
        ready_for_analysis = sum(1 for file in all_files if file.get("status") == "ready_for_analysis")
        
        # Collapsible Hypothesis Section
        with st.expander(f"Hypothesis ID: `{hypothesis_id}`", expanded=False):
            # Display Hypothesis Text
            st.write(f"**Hypothesis:** {hypothesis_text}")

            # File Status Summary
            st.write(f"**Files Attached:** {len(all_files)}")
            st.write(f"**Status:** `Processed: {processed_files}` | `Ready for Analysis: {ready_for_analysis}` | `Unprocessed: {unprocessed_files}`")
            
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
            
            # Download buttons
            st.write("**Analysis Downloads:**")
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