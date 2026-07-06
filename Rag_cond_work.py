# ==========================================================
# Conditional Workflow - RAG Setup
# ==========================================================

import os
from typing import TypedDict, Annotated

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
# ==========================================================

def build_rag_retriver(pdf_path: str):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    vectorStore = FAISS.from_documents(chunks, embedding)

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

# ==========================================================
# Define State
# ==========================================================

class State(TypedDict):
    Programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrived_context: str

# ==========================================================
# Classifier Node
# ==========================================================

def classifier_node(state: State) -> dict:
    """
    Look at the latest user message and classify the query type
    to decide which RAG retriever should be used.
    """

    last_message = state["messages"][-1]["content"]

    prompt = f"""
Classify the following student query into exactly one category:
'academic', 'fee', or 'general'.

Use 'academic' for questions about attendance, exams, grading, credits,
promotion, course structure, summer training, or degree requirements.

Use 'fee' for questions about tuition, payment, refund, late charges,
scholarships, or any money-related topic.

Use 'general' for greetings, casual talk, or anything not related to
the college rules or fee.

Query: {last_message}

Return only one word: academic, fee, or general.
"""

    response = LLM.invoke(prompt)

    category_query_type = response.content.strip().lower()

    if "academic" in category_query_type:
        category_query_type = "academic"
    elif "fee" in category_query_type:
        category_query_type = "fee"
    else:
        category_query_type = "general"

    return {
        "query_type": category_query_type
    }

# ==========================================================
# Academic Retriever Node
# ==========================================================

def Academic_rag_node_retriver(state: State) -> dict:
    """
    Retrieve context from the academics handbook.
    """

    query = state["messages"][-1]["content"]

    docs = academic_rag_retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    return {
        "retrived_context": context
    }

# ==========================================================
# Fee Retriever Node
# ==========================================================

def Fee_rag_node_retriver(state: State) -> dict:
    """
    Retrieve context from the fee structure document.
    """

    query = state["messages"][-1]["content"]

    docs = fee_retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    return {
        "retrived_context": context
    }

# ==========================================================
# General Retriever Node
# ==========================================================

def General_rag_node_retriver(state: State) -> dict:
    """
    For general queries, no retrieval is needed.
    """

    return {
        "retrived_context": "NO_RETRIEVAL_NEEDED"
    }

# ==========================================================
# Final Response Node
# ==========================================================

def final_node_reponse(state: State) -> dict:
    """
    Generate a final answer personalized using the student's programme.
    """

    query = state["messages"][-1]["content"]
    programme = state.get("Programme", "Unknown")
    context = state["retrived_context"]

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = f"""
You are a helpful college assistant talking to a {programme} student.

Answer the following query in a friendly and helpful manner using general knowledge.

Question:
{query}
"""

    else:

        prompt = f"""
You are a college assistant helping a {programme} student.

Use the following context from the official college documents to answer the question accurately.

If the context mentions specific figures for different programmes, highlight the one relevant to {programme} if possible.

Context:
{context}

Question:
{query}

Give a clear, friendly, and precise answer.
"""

    response = LLM.invoke(prompt)

    return {
        "messages": [("ai", response.content.strip())]
    }
    
    
    # router function to route to the appropriate retriever based on query type

def query_router_functions(state:State):
    if state["query_type"] == "academic":
        return Academic_rag_node_retriver(state)
    elif state["query_type"] == "fee":
        return Fee_rag_node_retriver(state)
    else:
        return General_rag_node_retriver(state)

# builing the graph for the conditional workflows

graph=StateGraph(
    state_type=State,
)

graph.add_node()