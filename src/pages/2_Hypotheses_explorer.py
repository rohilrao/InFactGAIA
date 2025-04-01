import streamlit as st
from pymongo import MongoClient
import gridfs
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId

# -------------------
# Streamlit Page Config
# -------------------
st.set_page_config(
    page_title="Hypothesis Explorer",
    layout="wide",  # "centered" also possible
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
# Orange Header (Markdown H2)
# -------------------
st.markdown(
    "<h2 style='color: orange; font-family: Arial, sans-serif;'>Hypothesis Explorer</h2>",
    unsafe_allow_html=True
)

# -------------------
# Minimalistic CSS
# -------------------
st.markdown(
    """
    <style>
    /* Overall text properties */
    body, [class^="st"], [data-testid="stForm"], [data-testid="stHeader"] {
        font-family: 'Arial', sans-serif;
    }

    /* Adjust expander header styling */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    /* Remove default expander borders */
    .stExpander {
        border: none !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }

    /* Hypothesis text styling */
    .hyp-text {
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    /* File list container */
    .file-container {
        padding: 0.5rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        max-height: 200px; /* scroll area limit */
        overflow-y: auto;  /* scroll overflow */
        background-color: #f9f9f9;
        border-radius: 4px;
    }

    /* Individual file item style */
    .file-item {
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
        display: flex;
        justify-content: space-between;
    }

    /* File name color */
    .file-name {
        color: #333333;
    }

    /* Status color */
    .status {
        color: #888888;
    }

    /* Section header within expanders */
    .section-header {
        font-size: 0.95rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #666666;
        font-weight: 600;
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
    st.write(f"**Total Hypotheses:** {len(hypotheses)}")

    for hypothesis in hypotheses:
        hypothesis_id = hypothesis["_id"]
        hypothesis_text = hypothesis["text"]
        
        # Exclude node_state and rendered_html files
        file_query = {
            "metadata.hypothesis_id": str(hypothesis_id),
            "filename": {"$not": {"$regex": "node_state|rendered_hypothesis"}}
        }
        all_files = list(db.fs.files.find(file_query))
        
        with st.expander(f"Hypothesis ID: {hypothesis_id}", expanded=False):
            st.markdown(
                f'<div class="hyp-text"><strong>Hypothesis:</strong> {hypothesis_text}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="hyp-text"><strong>Files Attached:</strong> {len(all_files)}</div>',
                unsafe_allow_html=True
            )
            
            # Display file list
            if all_files:
                st.markdown('<div class="file-container">', unsafe_allow_html=True)
                for file in all_files:
                    filename = file.get("filename", "unnamed")
                    status = file.get("status", "unknown")
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
            
            # Latest node state and rendered HTML
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
            st.markdown('<div class="section-header">Analysis Downloads</div>', unsafe_allow_html=True)
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
