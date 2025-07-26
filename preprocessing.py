from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class TextChunkerEmbedder:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def load_and_split(self, file_path):
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        return self.splitter.split_documents(docs)

    def embed_chunks(self, docs):
        texts = [doc.page_content for doc in docs]
        embeddings = self.embedding_model.embed_documents(texts)

        for i, doc in enumerate(docs):
            doc.metadata = doc.metadata or {}
            doc.metadata["embedding"] = embeddings[i]
            doc.metadata["doc_id"] = f"doc_{i}"
        return docs
