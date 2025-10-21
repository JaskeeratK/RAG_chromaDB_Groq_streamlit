# import os
# from langchain_community.document_loaders import PyPDFLoader, CSVLoader
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_core.runnables import Runnable
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os
import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import streamlit as st

# ✅ Use st.secrets instead
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)



# Custom prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant. Use the context to answer the question. Be concise.

Context:
{context}

Question:
{question}

Answer:
"""
)

# Load PDF or CSV
def load_docs(file_path: str):
    ext = file_path.split(".")[-1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError("Unsupported file type.")
    return loader.load()

# Create Chroma Vectorstore
from langchain_community.vectorstores import FAISS

def create_faiss_vectorstore_from_file(file_path):
    documents = load_docs(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    return vectorstore, embeddings


# Manual RAG Query
def query_rag_with_groq(query: str, vectorstore: Chroma, embedder) -> str:
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(final_prompt)
    return response.content


