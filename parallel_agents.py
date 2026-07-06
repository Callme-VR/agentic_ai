import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv

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
# Custom reducer function
# This merges the existing safety_score dictionary with
# new updates coming from multiple parallel nodes.
# =====================================================
def merger_score_dict(existing: dict, newUpdate: dict) -> dict:
    if existing is None:
        return newUpdate
    return {**existing, **newUpdate}

# =====================================================
# State Definition
# raw_text      -> Input text
# safety_score  -> Dictionary storing scores returned
#                  from different analyzer nodes
# =====================================================
class analyzer(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merger_score_dict]


# =====================================================
# Branch 1
# Safety Analyzer
# Checks profanity, toxicity, hate speech, etc.
# =====================================================
def safety_analyzer(state: analyzer) -> dict:
    print("=== SCAN: SAFETY ANALYZER 🤬 ===")

    prompt = f"""
You are an expert safety analyzer for profanity detection, abusive language,
aggression, hate speech toward any religion, race, gender, or individual,
and toxicity.

Analyze the following text and return ONLY a single integer score from 0 to 100.

0 means completely clean.
100 means extremely toxic.

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
# Branch 2
# Copyright Analyzer
# Checks plagiarism and trademark risk
# =====================================================
def Coyy_Analuzer(state: analyzer) -> dict:
    print("[Branch-2] Analyzing Copyright and Originality Risk 🤔...")

    prompt = f"""
Analyze the following text.

Judge whether it sounds heavily plagiarized,
unoriginal, or contains corporate trademark risks.

Provide a score from 0 to 100.

0 means completely original.
100 means very high copyright or trademark risk.

Return ONLY the integer number.

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
# Branch 3
# Cultural Analyzer
# Checks cultural and regional sensitivity
# =====================================================
def clutural_analyzer(state: analyzer) -> dict:
    print("[Branch-3] Analyzing Regional and Cultural Sensitivity 🌍...")

    prompt = f"""
Analyze the following text for cultural and regional sensitivity.

Determine whether the content could be offensive,
insensitive, or inappropriate for different cultures,
regions, religions, or communities.

Provide a score from 0 to 100.

0 means culturally safe.
100 means highly culturally insensitive.

Return ONLY the integer number.

Text:
{state["raw_text"]}
"""

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return the updated safety_score dictionary with sub-doctionary exact  under the `safety_score` key
    return {
        "safety_score": {
            "Cultural_sensitivity": score
        }
    }
    
builder=StateGraph(analyzer)