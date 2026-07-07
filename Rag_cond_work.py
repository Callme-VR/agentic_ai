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

def build_rag_retriever(pdf_path: str):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, embedding)

    return vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

# ==========================================================
# Create Retrievers
# ==========================================================

academic_rag_retriever = build_rag_retriever(
    pdf_path="academics_handbook.pdf"
)

fee_retriever = build_rag_retriever(
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
    last_message = state["messages"][-1].content

    prompt = f"""
Classify the following student query into exactly one category:
'academic', 'fee', or 'general'.

Use 'academic' for questions about attendance, exams, grading, credits,
promotion, course structure, summer training, or degree requirements.

Use 'fee' for questions about tuition, payment, refund, late charges,
scholarships, or any money-related topic.

Use 'general' for greetings, casual talk, or anything not related to
the college rules or fee.

Query:
{last_message}

Return only one word:
academic, fee, or general.
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

def Academic_rag_node_retriever(state: State) -> dict:
    query = state["messages"][-1].content

    docs = academic_rag_retriever.invoke(query)

    context = "\n".join(doc.page_content for doc in docs)

    return {
        "retrived_context": context
    }

# ==========================================================
# Fee Retriever Node
# ==========================================================

def Fee_rag_node_retriever(state: State) -> dict:
    query = state["messages"][-1].content  # FIXED: was ["content"]

    docs = fee_retriever.invoke(query)

    context = "\n".join(doc.page_content for doc in docs)

    return {
        "retrived_context": context
    }

# ==========================================================
# General Retriever Node
# ==========================================================

def General_rag_node_retriever(state: State) -> dict:
    return {
        "retrived_context": "NO_RETRIEVAL_NEEDED"
    }

# ==========================================================
# Final Response Node
# ==========================================================

def final_node_response(state: State) -> dict:
    query = state["messages"][-1].content  # FIXED: was ["content"]
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

Use the following context from the official college documents to answer accurately.

If the context contains information for multiple programmes,
highlight the answer relevant to {programme}.

Context:
{context}

Question:
{query}

Give a clear, friendly and precise answer.
"""

    response = LLM.invoke(prompt)

    return {
        "messages": [("ai", response.content.strip())]
    }

# ==========================================================
# Router Function
# ==========================================================

def query_router_function(state: State):

    if state["query_type"] == "academic":
        return "academic_rag"

    elif state["query_type"] == "fee":
        return "fee_rag"

    else:
        return "general_rag"

# ==========================================================
# Build Graph
# ==========================================================

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", Academic_rag_node_retriever)
graph.add_node("fee_rag", Fee_rag_node_retriever)
graph.add_node("general_rag", General_rag_node_retriever)
graph.add_node("final_response", final_node_response)

# ==========================================================
# Edges
# ==========================================================

graph.add_edge(START, "classifier")

graph.add_conditional_edges(
    "classifier",
    query_router_function
)

graph.add_edge("academic_rag", "final_response")
graph.add_edge("fee_rag", "final_response")
graph.add_edge("general_rag", "final_response")
graph.add_edge("final_response", END)

# ==========================================================
# Compile Graph
# ==========================================================

app = graph.compile()

# ==========================================================
# Run the App
# ==========================================================

print("Starting the RAG Conditional Workflow App...")
print("Which programme are you in?")
print("1. BCA")
print("2. B.Com")
print("3. BBA")

programme_map = {
    "1": "BCA",
    "2": "B.Com",
    "3": "BBA"
}

choice = input("Enter your choice (1-3): ")

student_programme = programme_map.get(choice, "BCA")

print(
    f"\nGreat! You are set as a {student_programme} student."
    "\nYou can now ask your questions about academics, fees, or general queries.\n"
)

while True:

    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit"]:
        print("Exiting the app. Goodbye!")
        break

    result = app.invoke(
        {
            "Programme": student_programme,
            "messages": [("human", user_query)]
        }
    )

    print(f"AI: {result['messages'][-1].content}\n")