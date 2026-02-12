import streamlit as st
import requests

st.set_page_config(page_title="CS 5542 Streamlit UI", layout="centered")

st.title("CS 5542 — Streamlit + FastAPI Demo")
st.caption("Type a message and send it to the FastAPI backend.")

API_URL = st.text_input("FastAPI base URL", value="http://127.0.0.1:3001")
text = st.text_input("Message")

if st.button("Send to API"):
    try:
        r = requests.post(f"{API_URL}/echo", json={"text": text}, timeout=10)
        r.raise_for_status()
        st.success("Response received")
        st.json(r.json())
    except Exception as e:
        st.error(f"Request failed: {e}")
