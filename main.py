import streamlit as st
from modules.pdf_reader import read_pdf
from modules.chunker import chunk_text
from modules.embedder import get_embedding
from modules.retriever import Retriever
from modules.agent import Agent

st.set_page_config(page_title="Agentic RAG with Gemini", layout="wide")
st.title("Agentic RAG with Gemini")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF and splitting into chunks..."):
        text = read_pdf(uploaded_file)
        chunks = chunk_text(text)
        st.success(f"PDF processed into {len(chunks)} chunks.")

    with st.spinner("Initializing Agent..."):
        embeddings = [get_embedding(chunk) for chunk in chunks]
        retriever = Retriever(chunks, embeddings)
        agent = Agent(retriever)

    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("Agent is thinking..."):
            answer = agent.ask(question)
            st.subheader("Answer")
            st.write(answer)
