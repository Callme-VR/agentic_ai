import os
from dotenv import load_dotenv
load_dotenv()  # Loads GROQ_API_KEY, MISTRAL_API_KEY, TAVILY_API_KEY etc. from a .env file

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  # Reducer: appends new messages to the list instead of overwriting
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import ToolNode

# =========================================================
# TOOLS
# =========================================================
# Tavily is a web-search tool. The writer LLM can call this
# when it needs fresh/current info to write an accurate post.
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# ToolNode is a prebuilt LangGraph node that automatically executes
# whichever tool(s) the LLM decided to call (based on tool_calls in
# the last AIMessage) and returns ToolMessages with the results.
tool_node = ToolNode(tools)

# =========================================================
# LLMs
# =========================================================
# Writer model: generates/rewrites the LinkedIn post.
# Slightly higher temperature (0.7) for more creative, natural writing.
writter_llm = ChatMistralAI(
    model="mistral-medium-latest",  # verify exact slug in Mistral docs
    temperature=0.7,
    api_key=os.getenv("MISTRAL_API_KEY"),
)
# bind_tools() lets this LLM decide (on its own) whether to call
# the search tool before answering, by emitting a tool_calls list.
writter_llm_with_tools = writter_llm.bind_tools(tools)

# Reviewer model: judges the draft against fixed criteria.
# Lower temperature (0.3) for more consistent, deterministic grading.
review_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Hard cap on writer<->reviewer loops so the graph can't run forever
# if the reviewer keeps rejecting.
MAX_ATTEMPTS = 3

# =========================================================
# STATE
# =========================================================
# This TypedDict defines the shared "memory" that flows between
# every node in the graph. Each node returns a partial dict, and
# LangGraph merges it into this state (using the reducer where defined,
# e.g. add_messages for "messages").
class State(TypedDict):
    topic: str                     # user-provided topic for the post
    messages: Annotated[list, add_messages]  # full LLM/tool conversation history (auto-appended)
    draft: str                     # latest generated post text
    review_feedback: str           # reviewer's feedback on the latest draft
    isApproved: bool               # whether the reviewer approved the draft
    attempt: int                   # how many times the writer has run

# =========================================================
# WRITER NODE
# =========================================================
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
    """
    Generates a new draft on attempt 1, or rewrites the previous draft
    using the reviewer's feedback on later attempts.
    May also emit tool_calls (to trigger a web search) instead of
    a finished draft — that's handled by should_use_tool().
    """
    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    if attempt == 1:
        # First pass: plain instruction, optionally search the web.
        user_message = (
            f"Write a LinkedIn post on this topic: {topic}\n"
            f"If you need current info, search the web first using tavily search."
        )
    else:
        # Retry pass: explicitly feed back the reviewer's critique so
        # the model fixes the specific issues instead of starting blind.
        user_message = (
            f"Your previous draft's topic: '{topic}'\n"
            f"Review feedback: {previous_feedback}\n"
            f"Rewrite the post based on the feedback — write a new, improved draft that fixes every issue mentioned."
        )

    # Note: we send a fresh [system, human] pair each time rather than
    # the full accumulated history — keeps the prompt short and focused.
    messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]
    response = writter_llm_with_tools.invoke(messages)

    return {
        # These get appended to state["messages"] via the add_messages reducer,
        # so the tool node (if triggered) can see the human prompt + AI tool call.
        "messages": [("human", user_message), response],
        "attempt": attempt,
    }

def should_use_tool(state: State) -> str:
    """
    Routing function after the writer node.
    If the writer's last response contains tool_calls (it wants to search
    the web), route to the "tools" node. Otherwise, the response is a
    finished draft, so move on to extracting it.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "extract_draft"

# =========================================================
# EXTRACT NODE
# =========================================================
def extract_draft_node(state: State) -> dict:
    """
    Pulls the plain text of the writer's final (non-tool-call) response
    out of the messages list and stores it as the current draft.
    """
    last_message = state["messages"][-1]
    draft = last_message.content
    print(f"\n\nGenerated post:\n{draft}\n\n")
    return {"draft": draft}

# =========================================================
# REVIEWER NODE
# =========================================================
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
    """
    Sends the current draft to the reviewer LLM and parses a
    structured VERDICT / FEEDBACK response out of the plain text.
    """
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

    # Naive parsing: look for "APPROVED" only in the text BEFORE the
    # "FEEDBACK" section, so the word "approved" mentioned inside the
    # feedback paragraph itself can't accidentally flip the verdict.
    is_approved = "APPROVED" in review_text.upper().split("FEEDBACK")[0]

    if "FEEDBACK:" in review_text:
        feedback = review_text.split("FEEDBACK:", 1)[1].strip()
    else:
        # Fallback in case the model didn't follow the exact format.
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"
    print(f"[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return {"review_feedback": feedback, "isApproved": is_approved}

def should_stop_looping_review(state: State) -> str:
    """
    Routing function after the reviewer node.
    Ends the graph if the post is approved OR the attempt limit is hit;
    otherwise loops back to the writer to try again with feedback.
    """
    if state["isApproved"]:
        print("Post has been approved.")
        return END
    if state["attempt"] >= MAX_ATTEMPTS:
        print("Post exceeded the maximum number of attempts.")
        return END
    return "writer"

# =========================================================
# BUILD GRAPH
# =========================================================
# Flow:
# START -> writer -> (tools -> writer)* -> extract_draft -> reviewer -> (writer | END)
graph = StateGraph(State)

graph.add_node("writer", writer_node_post)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node_post)

graph.add_edge(START, "writer")

# After the writer runs, decide whether it needs a web search
# (-> tools) or has produced a final draft (-> extract_draft).
graph.add_conditional_edges(
    "writer",
    should_use_tool,
    {"tools": "tools", "extract_draft": "extract_draft"},
)

# Tool results always go back to the writer so it can incorporate
# the search results into (or finish) the draft.
graph.add_edge("tools", "writer")          # tool result goes back to writer, not reviewer

# Once a draft is extracted, it always goes to the reviewer.
graph.add_edge("extract_draft", "reviewer")

# After review, either loop back to the writer with feedback, or end.
graph.add_conditional_edges(
    "reviewer",
    should_stop_looping_review,
    {"writer": "writer", END: END},
)

app = graph.compile()

# =========================================================
# RUN (CLI loop)
# =========================================================
if __name__ == "__main__":
    while True:
        topic = input("Enter the topic for your LinkedIn post (or 'exit' to quit): ").strip()
        if topic.lower() == "exit":
            break

        # Kick off the graph with a fresh state for this topic.
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