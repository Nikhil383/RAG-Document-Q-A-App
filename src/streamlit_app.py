import streamlit as st
import requests

API = "http://localhost:8000"

st.title("RAG System — Upload Documents (PDF/TXT/DOCX) + Ask Questions")

# Upload file section
file = st.file_uploader("Upload a document", type=["pdf", "txt", "md", "docx", "doc"])

if file and st.button("Upload & Index Document"):
    with st.spinner("Uploading & indexing..."):
        resp = requests.post(API + "/upload", files={"file": (file.name, file.getvalue())})
        st.write(resp.json())

st.divider()

# Ask a question
question = st.text_input("Ask a question about your uploaded documents:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Write a question.")
    else:
        resp = requests.post(API + "/answer", json={"question": question})
        data = resp.json()

        st.subheader("Answer:")
        st.write(data["answer"])

        st.subheader("Sources:")
        for s in data.get("sources", []):
            st.write(s)
