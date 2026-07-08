import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import ToolNode

# ---------- tools ----------
search_tool = TavilySearch(max_results=3)
tools = [search_tool]
tool_node = ToolNode(tools)

# ---------- LLMs ----------
writter_llm = ChatMistralAI(
    model="mistral-medium-latest",  # verify exact slug in Mistral docs
    temperature=0.7,
    api_key=os.getenv("MISTRAL_API_KEY"),
)
writter_llm_with_tools = writter_llm.bind_tools(tools)

review_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

MAX_ATTEMPTS = 3

# ---------- state ----------
class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    isApproved: bool
    attempt: int

# ---------- writer ----------
WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)

def writer_node_post(state: State) -> dict:
    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic: {topic}\n"
            f"If you need current info, search the web first using tavily search."
        )
    else:
        user_message = (
            f"Your previous draft's topic: '{topic}'\n"
            f"Review feedback: {previous_feedback}\n"
            f"Rewrite the post based on the feedback — write a new, improved draft that fixes every issue mentioned."
        )

    messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]
    response = writter_llm_with_tools.invoke(messages)

    return {
        "messages": [("human", user_message), response],
        "attempt": attempt,
    }

def should_use_tool(state: State) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "extract_draft"

# ---------- extract ----------
def extract_draft_node(state: State) -> dict:
    last_message = state["messages"][-1]
    draft = last_message.content
    print(f"\n\nGenerated post:\n{draft}\n\n")
    return {"draft": draft}

# ---------- reviewer ----------
REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)

def reviewer_node_post(state: State) -> dict:
    draft = state["draft"]

    prompt = (
        f"Review this LinkedIn post draft:\n"
        f"{draft}\n"
        f"Give your review in the format specified."
    )

    response = review_llm.invoke(
        [
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", prompt),
        ]
    )

    review_text = response.content.strip()
    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return {"review_feedback": feedback, "isApproved": is_approved}

def should_stop_looping_review(state: State) -> str:
    if state["isApproved"]:
        print("Post has been approved.")
        return END
    if state["attempt"] >= MAX_ATTEMPTS:
        print("Post exceeded the maximum number of attempts.")
        return END
    return "writer"

# ---------- build graph ----------
graph = StateGraph(State)

graph.add_node("writer", writer_node_post)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node_post)

graph.add_edge(START, "writer")
graph.add_conditional_edges(
    "writer",
    should_use_tool,
    {"tools": "tools", "extract_draft": "extract_draft"},
)
graph.add_edge("tools", "writer")          # tool result goes back to writer, not reviewer
graph.add_edge("extract_draft", "reviewer")
graph.add_conditional_edges(
    "reviewer",
    should_stop_looping_review,
    {"writer": "writer", END: END},
)

app = graph.compile()

# ---------- run ----------

while True:
    topic = input("Enter the topic for your LinkedIn post (or 'exit' to quit): ").strip()
    if topic.lower() == "exit":
        break

    result = app.invoke({
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "isApproved": False,
        "attempt": 0,
    })

    print("\n=== FINAL POST ===\n")
    print(result["draft"])
    print("Approved:", result["isApproved"])
