import streamlit as st
from pymongo import MongoClient
import gridfs
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId

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
fs = gridfs.GridFS(db)  # For file storage
hypothesis_collection = db["hypotheses"]

def ensure_object_id(id_value):
    """Convert string IDs to ObjectId if needed."""
    if isinstance(id_value, str) and ObjectId.is_valid(id_value):
        try:
            return ObjectId(id_value)
        except Exception:
            pass
    return id_value

# -------------------
# Page Header
# -------------------
st.markdown("### :orange[Hypothesis Explorer]")

# -------------------
# Minimal CSS (layout only; color handled inline below)
# -------------------
st.markdown(
    """
    <style>
    /* General font family */
    body, [class^="st"], [data-testid="stForm"], [data-testid="stHeader"] {
        font-family: 'Arial', sans-serif;
    }

    /* Make expander content neat */
    .stExpander {
        border: none !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }

    /* Scrollable container for files */
    .file-container {
        background-color: #f9f9f9;
        border-radius: 4px;
        max-height: 200px;
        overflow-y: auto;
        padding: 0.5rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Slight spacing for each file item */
    .file-item {
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------
# Hypothesis Explorer Logic
# -------------------
hypotheses = list(hypothesis_collection.find({}))

if hypotheses:
    st.markdown(f"**Total Hypotheses:** {len(hypotheses)}")

    for hypothesis in hypotheses:
        hypothesis_id = hypothesis["_id"]
        hypothesis_text = hypothesis["text"]
        
        # Exclude node_state and rendered_hypothesis files
        file_query = {
            "metadata.hypothesis_id": str(hypothesis_id),
            "filename": {"$not": {"$regex": "node_state|rendered_hypothesis"}}
        }
        all_files = list(db.fs.files.find(file_query))
        
        # Expander for each hypothesis
        with st.expander(f"Hypothesis ID: {hypothesis_id}", expanded=False):
            st.markdown(f"**:orange[Hypothesis]:** {hypothesis_text}")
            st.markdown(f"**:orange[Files Attached]:** {len(all_files)}")
            
            # Display files if present
            if all_files:
                st.markdown('<div class="file-container">', unsafe_allow_html=True)
                for file in all_files:
                    filename = file.get("filename", "unnamed")
                    status = file.get("status", "unknown")
                    st.markdown(
                        f'<div class="file-item">'
                        f':blue[{filename}] — :green[{status}]'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No files attached to this hypothesis.")
            
            # Retrieve latest node_state and HTML
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

            # Analysis Downloads
            st.markdown("### :orange[Analysis Downloads]")
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
