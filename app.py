# app.py
import streamlit as st
from rag_core import create_faiss_vectorstore_from_file, query_rag_with_groq
import tempfile
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("Multimodal RAG Chatbot")

# Session state for uploaded file and chatbot
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.embedder = None
    st.session_state.filename = None

# Upload Section
uploaded_file = st.file_uploader("📄 Upload a CSV or PDF file", type=["csv", "pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        st.session_state.vectorstore, st.session_state.embedder = create_faiss_vectorstore_from_file(tmp_path)
        st.session_state.filename = uploaded_file.name
        st.success(f"✅ Vector store created for `{uploaded_file.name}`.")
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")

# Chat Section
st.divider()
st.subheader("💬 Ask a question about the uploaded document")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask your question here...")

if user_query and st.session_state.vectorstore:
    with st.spinner("Querying..."):
        reply = query_rag_with_groq(user_query, st.session_state.vectorstore, st.session_state.embedder)
    st.session_state.chat_history.append(("You", user_query))
    st.session_state.chat_history.append(("Bot", reply))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(msg)
