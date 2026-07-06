import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# =====================================================
# Load environment variables from .env file
# =====================================================
load_dotenv()

# =====================================================
# Initialize the Groq LLM
# =====================================================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# =====================================================
# Reducer Function
# Merges dictionaries returned by parallel nodes
# =====================================================
def merger_score_dict(existing: dict, newUpdate: dict) -> dict:
    if existing is None:
        return newUpdate
    return {**existing, **newUpdate}

# =====================================================
# State Definition
# =====================================================
class analyzer(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merger_score_dict]


# =====================================================
# Branch 1 : Toxicity Analyzer
# =====================================================
def Toxicity_safety_analyzer(state: analyzer) -> dict:
    print("=== SCAN: SAFETY ANALYZER 🤬 ===")

    prompt = f"""
You are an expert safety analyzer.

Analyze the following text for:
- Profanity
- Hate Speech
- Abusive Language
- Aggression
- Toxicity

Return ONLY one integer between 0 and 100.

0 = Completely Safe
100 = Extremely Toxic

Text:
{state["raw_text"]}
"""

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {
        "safety_score": {
            "Toxicity_level": score
        }
    }


# =====================================================
# Branch 2 : Copyright Analyzer
# =====================================================
def Copyright_Analyzer(state: analyzer) -> dict:
    print("=== SCAN: COPYRIGHT ANALYZER 📄 ===")

    prompt = f"""
Analyze the following text.

Determine whether it appears:

- Plagiarized
- Unoriginal
- Contains trademark risks

Return ONLY one integer between 0 and 100.

0 = Completely Original
100 = High Copyright Risk

Text:
{state["raw_text"]}
"""

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {
        "safety_score": {
            "Copyright_level": score
        }
    }


# =====================================================
# Branch 3 : Cultural Analyzer
# =====================================================
def Cultural_Analyzer(state: analyzer) -> dict:
    print("=== SCAN: CULTURAL ANALYZER 🌍 ===")

    prompt = f"""
Analyze the following text.

Determine whether it is culturally insensitive,
offensive toward any community,
religion, race or region.

Return ONLY one integer between 0 and 100.

0 = Completely Safe
100 = Highly Offensive

Text:
{state["raw_text"]}
"""

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    return {
        "safety_score": {
            "Cultural_sensitivity": score
        }
    }


# =====================================================
# Build LangGraph Workflow
# =====================================================

Builder_graph = StateGraph(analyzer)

# -----------------------------------------------------
# Register Nodes
# -----------------------------------------------------

Builder_graph.add_node("safety_analyzer", Toxicity_safety_analyzer)
Builder_graph.add_node("copyright_analyzer", Copyright_Analyzer)
Builder_graph.add_node("cultural_analyzer", Cultural_Analyzer)

# -----------------------------------------------------
# Fan-Out
# START -> Three Parallel Nodes
# -----------------------------------------------------

Builder_graph.add_edge(START, "safety_analyzer")
Builder_graph.add_edge(START, "copyright_analyzer")
Builder_graph.add_edge(START, "cultural_analyzer")

# -----------------------------------------------------
# Fan-In
# Three Nodes -> END
# -----------------------------------------------------

Builder_graph.add_edge("safety_analyzer", END)
Builder_graph.add_edge("copyright_analyzer", END)
Builder_graph.add_edge("cultural_analyzer", END)

# -----------------------------------------------------
# Compile Graph
# -----------------------------------------------------

app = Builder_graph.compile()

# =====================================================
# Sample Input
# =====================================================

simple_script = """
Yo guys! Welcome back to the stream.

Today I am going to show you how to hack into your friend's
system using some script I copied directly from online forums.

Honestly, traditional security protocols are absolute garbage
and anyone still using them is an absolute idiot.

Let's dive into the code and fuck this system.
"""

# =====================================================
# Execute Workflow
# =====================================================

result = app.invoke(
    {
        "raw_text": simple_script
    }
)

# =====================================================
# Print Results
# =====================================================

print("\n" + "=" * 60)
print("FINAL SAFETY REPORT")
print("=" * 60)

print(result["safety_score"])

print("=" * 60)



# what is Reducers in langgraph
