import streamlit as st
from pymongo import MongoClient
import gridfs
import datetime
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi

# 🔐 MongoDB Connection
@st.cache_resource
def get_db_client():
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

client = get_db_client()
db = client["hypothesis_management"]
fs = gridfs.GridFS(db)
hypothesis_collection = db["hypotheses"]

# Function to query the MongoDB for hypothesis ID
def query_hypothesis(hypothesis_id):
    try:
        hypothesis = hypothesis_collection.find_one({"_id": ObjectId(hypothesis_id)})
        if hypothesis:
            return hypothesis
        else:
            st.error("Hypothesis not found")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app layout
st.title("Result Explorer")

# Input for Hypothesis ID
hypothesis_id = st.text_input("Enter Hypothesis ID:")

if hypothesis_id:
    data = query_hypothesis(hypothesis_id)
    if data:
        st.subheader("Associated Files")
        for file in data.get("files", []):
            st.write(file)

        st.subheader("Results")
        for result in data.get("results", []):
            st.write(result)