import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. Load the environment variables directly from .env
load_dotenv()

# Ensure the key exists without prompting the terminal
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from your .env file!")

# 2. Initialize LLM
GROQ_MODEL = "llama-3.1-8b-instant"
llm = ChatGroq(model=GROQ_MODEL, temperature=0)

# 3. Define State Schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. Define Node Logic
def chatnode(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# 5. Construct Graph Layout
graph = StateGraph(ChatState)
graph.add_node('chat_node', chatnode)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

# 6. Memory Checkpointer
memory = MemorySaver()
chatbot = graph.compile(checkpointer=memory)