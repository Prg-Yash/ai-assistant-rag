# ui.py
import streamlit as st
import requests

st.set_page_config(page_title="Real-time Assistant (Pathway + Gemini)", layout="centered")
st.title("🌐 Real-time Assistant (Pathway + Gemini)")

server_url = st.text_input("Pathway server URL", "http://localhost:8000/query")
query = st.text_input("Ask a question (e.g. 'Latest bitcoin news?'):")

if st.button("Ask") and query.strip():
    try:
        r = requests.post(server_url, json={"query": query}, timeout=10)
        r.raise_for_status()
        data = r.json()
        st.subheader("Raw response")
        st.json(data)
        # If response contains 'answer' show it:
        if "answer" in data:
            st.subheader("Answer")
            st.write(data["answer"])
    except Exception as e:
        st.error(f"Request failed: {e}")
