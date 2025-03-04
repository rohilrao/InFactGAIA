import streamlit as st

st.set_page_config(page_title="InFactGAIA", page_icon="📂", layout="wide")

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Hypothesis Explorer"])

if page == "Home":
    st.title("🏠 Welcome to InFactGAIA")
    st.write("Use the sidebar to navigate through the app.")

elif page == "Hypothesis Explorer":
    from pages.hypothesis_explorer import hypothesis_explorer
    hypothesis_explorer()



