import os
from typing_extensions import TypedDict, NotRequired

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

# ==========================================================
# State Definition
# ==========================================================

class Pipelines(TypedDict):
    raw_input: str
    editor_text: NotRequired[str]
    script_text: NotRequired[str]
    final_output: NotRequired[str]


# ==========================================================
# LLM Configuration
# ==========================================================

LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

# ==========================================================
# Function: editor_node()
# Stage 1 - Clean the raw text
# ==========================================================

def editor_node(state: Pipelines) -> dict:
    print("\n========== EXECUTING EDITOR NODE ==========\n")

    prompt = f"""
You are an expert copy editor.

Your task is to:
1. Correct grammatical mistakes.
2. Fix spelling mistakes and typos.
3. Improve sentence flow.
4. Preserve the original meaning.
5. Return only the edited text.

Text:
{state["raw_input"]}
"""

    response = LLM.invoke(prompt)

    print("Editor Finished ✅")

    return {
        "editor_text": response.content.strip()
    }


# ==========================================================
# Function: script_writer()
# Stage 2 - Convert into YouTube script
# ==========================================================

def script_writer(state: Pipelines) -> dict:
    print("\n========== EXECUTING SCRIPT WRITER ==========\n")

    prompt = f"""
You are a professional YouTube content creator.

Convert the following edited text into a highly engaging,
conversational YouTube script.

Rules:
- Begin with a strong hook.
- Keep the audience engaged.
- Use a friendly tone.
- End with a memorable conclusion.
- Return only the script.

Edited Text:

{state["editor_text"]}
"""

    response = LLM.invoke(prompt)

    print("Script Writer Finished ✅")

    return {
        "script_text": response.content.strip()
    }


# ==========================================================
# Function: hinglish_converter()
# Stage 3 - Convert to Hinglish
# ==========================================================

def hinglish_converter(state: Pipelines) -> dict:
    print("\n========== EXECUTING HINGLISH CONVERTER ==========\n")

    prompt = f"""
You are an expert translator.

Convert the following YouTube script into natural Hinglish.

Rules:
- Mix Hindi and English naturally.
- Don't translate technical words unnecessarily.
- Make it sound conversational.
- Keep the meaning exactly the same.
- Return only the Hinglish script.

Script:

{state["script_text"]}
"""

    response = LLM.invoke(prompt)

    print("Hinglish Conversion Finished ✅")

    return {
        "final_output": response.content.strip()
    }


# ==========================================================
# Build LangGraph Workflow
# ==========================================================

graph = StateGraph(Pipelines)

# Add Nodes
graph.add_node("editor", editor_node)
graph.add_node("script_writer", script_writer)
graph.add_node("hinglish_converter", hinglish_converter)

# Add Edges
graph.add_edge(START, "editor")
graph.add_edge("editor", "script_writer")
graph.add_edge("script_writer", "hinglish_converter")
graph.add_edge("hinglish_converter", END)

# Compile Graph
app = graph.compile()


# ==========================================================
# Run the Workflow
# ==========================================================

if __name__ == "__main__":

    result = app.invoke(
        {
            "raw_input": (
                "AI is transforming the world where AI agents can think, "
                "plan, book flight tickets, make hotel reservations, "
                "manage calendars, and automate many daily tasks for humans."
            )
        }
    )

    print("\n" + "=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)
    print(result["final_output"])
    print("=" * 70)