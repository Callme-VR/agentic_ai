import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from typing import TypedDict ,Annotated
from langgraph.graph import StateGraph ,START ,END
from langgraph.graph.message import add_messages
from langchain_mistralai import ChatMistralAI
from langraph.prebuilt import ToolNode
# made the tools
search_tool=TavilySearch(
     max_results=3
)

tools=[search_tool]


# llm for writter the post with  more creavity 

writter_llm=ChatMistralAI(
     model="Mistral-Medium-3.5",
     temperature=0.7,
     api_key=os.getenv("MISTRAL_API_KEY"),
     # for more creativity post of the linkdin
)

# bind with tools 
writter_llm_with_tools=writter_llm.bind_tools(tools)


# for review the post review_llm

review_llm=ChatGroq(
     model="llama-3.3-70b-versatile",
     temperature=0.3,
     api_key=os.getenv("GROQ_API_KEY"),
     # for more creativity post of the linkdin
)


# states building the 

class State(TypedDict):
     topic:str
     messages: Annotated[list, add_messages]
     draft: str
     review_feedback: str
     isApproved:bool
     attempt:int
     
     
     
     
# nodes
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



# first node
def writer_node_post(state:State)->dict:
  """
  Write a new LinkedIn post or rewrite an existing one.
  If needed, first search the web (using Tavily) to gather relevant,
  up-to-date information, trends, or supporting facts before generating
  the final LinkedIn post.
  """
  
  