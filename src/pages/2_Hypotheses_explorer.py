import streamlit as st
from pymongo import MongoClient
import gridfs
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
import pandas as pd

# -------------------
# Streamlit Page Config
# -------------------
st.set_page_config(
    page_title="InFactGAIAV3",
    page_icon="📂",
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
        
        # Exclude node_state and rendered_hypothesis
        file_query = {
            "metadata.hypothesis_id": str(hypothesis_id),
            "filename": {"$not": {"$regex": "node_state|rendered_hypothesis"}}
        }
        all_files = list(db.fs.files.find(file_query))
        
        with st.expander(f"Hypothesis ID: {hypothesis_id}", expanded=False):
            # Highlight the hypothesis text in orange
            st.markdown(f"**Hypothesis:** :orange[{hypothesis_text}]")
            
            # Simple count of files
            st.markdown(f"**Files Attached:** {len(all_files)}")
            if all_files:
                # Create scrollable container for files
                with st.container():
                    # Set a fixed height for scrolling with CSS
                    st.markdown(
                        """
                        <style>
                            .file-list {
                                max-height: 200px;
                                overflow-y: auto;
                                padding: 10px;
                                border: 1px solid #e6e6e6;
                                border-radius: 5px;
                            }
                        </style>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Start the scrollable container
                    file_list_html = '<div class="file-list">'
                    
                    # Add each file to the HTML content
                    for file in all_files:
                        filename = file.get("filename", "unnamed")
                        status = file.get("status", "unknown")
                        file_list_html += f"<p><strong>{filename}</strong> → <em>{status}</em></p>"
                    
                    # Close the container
                    file_list_html += '</div>'
                    
                    # Render the entire HTML at once
                    st.markdown(file_list_html, unsafe_allow_html=True)
            else:
                st.info("No files attached to this hypothesis.")
            # Latest node state & rendered HTML
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
            
            # Download Buttons
            #st.write("**Analysis Downloads**")
            col1, col2 = st.columns(2)
            
            with col1:
                if latest_node_state:
                    node_state_data = fs.get(ensure_object_id(latest_node_state["_id"])).read()
                    st.download_button(
                        label="Download Node State (JSON)",
                        data=node_state_data,
                        file_name=f"hypothesis_{hypothesis_id}_state.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.info("No node state available")
            
            with col2:
                if latest_html:
                    html_data = fs.get(ensure_object_id(latest_html["_id"])).read()
                    st.download_button(
                        label="Download Visualization (HTML)",
                        data=html_data,
                        file_name=f"hypothesis_{hypothesis_id}_visualization.html",
                        mime="text/html",
                        use_container_width=True
                    )
                else:
                    st.info("No visualization available")

else:
    st.warning("No hypotheses found in the database.")
