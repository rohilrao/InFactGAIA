import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import gridfs

@st.cache_resource
def get_db_client():
    """
    Creates and caches a MongoDB client connection.
    Returns the client object.
    """
    MONGO_URI = st.secrets["MONGO_URI"]
    return MongoClient(MONGO_URI, server_api=ServerApi("1"))

def initialize_db():
    """
    Initializes database connections and returns relevant database objects.
    
    Returns:
        tuple: (client, db, fs, hypothesis_collection)
    """
    client = get_db_client()
    db = client["infact_db_v3"]
    fs = gridfs.GridFS(db)
    hypothesis_collection = db["hypotheses"]
    
    return client, db, fs, hypothesis_collection