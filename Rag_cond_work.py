# conditional -workflow 

import os
from typing import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import langchain_huggingface
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# build the simple rag for the Conditional-workflow-langgraph

def build_rag_retriver(
    pdf_path:str
):
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()
    
    # text splitter
    
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    
    # chunks
    chunks=splitter.split_documents(documents=documents)
    vectorStore=FAISS.from_documents(chunks,embedding)
    return vectorStore.as_retriever(search_kwargs={"k":3})
    