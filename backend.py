import os
from typing import Annotated, TypedDict
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_tavily import TavilySearch
import requests
import math


load_dotenv()
connection = sqlite3.connect("langraph_chat.db",check_same_thread=False)
print("LangSmith Tracing Enabled:", os.getenv("LANGSMITH_TRACING"))


if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from your .env file!")


@tool
def calculator(expression: str) -> str:
    """Useful for evaluating mathematical expressions. 
    Input should be a clear mathematical string like '2 * (3 + 5)' or 'math.sqrt(144)'."""
    try:
        allowed_names = {"math": __import__("math")}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
@tool
def get_stock_price(ticker: str) -> str:
    """Retrieves the real-time stock price for a given stock ticker symbol (e.g., AAPL, GOOG, MSFT)."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Error: ALPHA_VANTAGE_API_KEY is missing from environment variables."
    
    # 1. Clean the ticker input (remove spaces and make uppercase)
    clean_ticker = ticker.strip().upper()
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={clean_ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        
        # 2. Check for Rate Limit messages first
        if "Information" in data:
            return f"API Rate Limit Hit: {data['Information']}"
        
        quote = data.get("Global Quote", {})
        if not quote:
            return f"Could not find stock data for ticker '{clean_ticker}'. Make sure it's a valid US equity ticker symbol (like AAPL for Apple)."
            
        price = quote.get("05. price", "N/A")
        change = quote.get("09. change", "N/A")
        change_percent = quote.get("10. change percent", "N/A")
        
        return f"Stock: {clean_ticker} | Current Price: ${price} | Change: {change} ({change_percent})"
    except Exception as e:
        return f"An error occurred while fetching stock data: {str(e)}"

web_search = TavilySearch(max_results=5)

# 2. Initialize LLM
GROQ_MODEL = "llama-3.1-8b-instant"
llm = ChatGroq(model=GROQ_MODEL, temperature=0)
tools = [calculator, get_stock_price, web_search]
llm_tools = llm.bind_tools(tools)
# 3. Define State Schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. Define Node Logic
def chatnode(state: ChatState):
    messages = state['messages']
    response = llm_tools.invoke(messages)
    return {"messages": [response]}

# 5. Construct Graph Layout
tool_node = ToolNode(tools)
graph = StateGraph(ChatState)
graph.add_node('chat_node', chatnode)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges(
    'chat_node',
    tools_condition,
)
graph.add_edge('tools', 'chat_node')
memory = SqliteSaver(connection)
chatbot = graph.compile(checkpointer=memory)


def get_threads():
    all_threads = set()
    threads = memory.list(None)
    for thread in threads:
        all_threads.add(thread.config['configurable']['thread_id'])
    return list(all_threads)

