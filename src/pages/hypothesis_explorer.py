import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# 🔐 MongoDB Connection
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["hypothesis_management"]
hypothesis_collection = db["hypotheses"]

# 📌 Hypothesis Explorer Page
st.title("📜 Hypothesis Explorer")

# 📡 Fetch all stored hypotheses
hypotheses = list(hypothesis_collection.find({}))

if hypotheses:
    st.write(f"📊 **Total Hypotheses Stored:** {len(hypotheses)}")

    # 📌 Display each Hypothesis ID and Text
    for hypothesis in hypotheses:
        with st.expander(f"🆔 {hypothesis['_id']}"):
            st.write(f"**Hypothesis Text:**\n\n{hypothesis['text']}")

else:
    st.warning("⚠️ No hypotheses found in the database.")
