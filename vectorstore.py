import os
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from config import DATA_PATH, CHROMA_PATH

# Load documents from PDF directory
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Split documents into chunks for vector storage
def split_documents(documents, max_tokens=1000, token_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=token_overlap)
    return text_splitter.split_documents(documents)

# Create or load a vector store using Chroma
def create_or_load_vector_store(embedding_model):
    documents = load_documents()
    chunks = split_documents(documents)

    embedding_function = OllamaEmbeddings(model=embedding_model)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Add documents to vector store if it's new
    db.add_documents(chunks)
    db.persist()

    return db
