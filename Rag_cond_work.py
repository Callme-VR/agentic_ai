# ==========================================================
# Conditional Workflow - RAG Setup
# ==========================================================

import os
from typing import TypedDict,Annotated

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

# ==========================================================
# Initialize HuggingFace Embedding Model
# Used to convert text chunks into vector embeddings
# ==========================================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================================
# Initialize Groq LLM
# ==========================================================

LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
)

# ==========================================================
# Build a Simple RAG Retriever
# Loads PDF
# Splits into chunks
# Creates FAISS Vector Store
# Returns Retriever
# ==========================================================

def build_rag_retriver(pdf_path: str):

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the document into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    # Create document chunks
    chunks = splitter.split_documents(documents)

    # Create FAISS Vector Store
    vectorStore = FAISS.from_documents(chunks, embedding)

    # Return Retriever
    return vectorStore.as_retriever(
        search_kwargs={"k": 4}
    )

# ==========================================================
# Create Retrievers
# ==========================================================

academic_rag_retriever = build_rag_retriver(
    pdf_path="academics_handbook.pdf"
)

fee_retriever = build_rag_retriver(
    pdf_path="fee_structure.pdf"
)

# =========================================================
# Define  states    TypedDict for RAG Retriever Mapping
# =========================================================
class State(TypedDict):
    Programme: str
    messages: Annotated[list, add_messages]
    query_type:str
    retrived_context:str
    
# step-3 Nodes genrations
