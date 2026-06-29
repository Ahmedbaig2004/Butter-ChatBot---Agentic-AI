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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt


import streamlit as st

import requests
import math


load_dotenv()
connection = sqlite3.connect("langraph_chat.db",check_same_thread=False)
print("LangSmith Tracing Enabled:", os.getenv("LANGSMITH_TRACING"))

def initialize_dynamic_rag(file_path: str):
    """Loads a specific PDF, chunks it, and returns a FAISS retriever."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pdf_vectorstore = FAISS.from_documents(pdf_chunks, embeddings)
    return pdf_vectorstore.as_retriever(search_kwargs={"k": 2})

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing from your .env file!")
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from your .env file!")


pdf_retrievers: dict[str, any] = {}
def set_pdf_retriever(thread_id: str, retriever):
    pdf_retrievers[thread_id] = retriever
@tool
def rag(query: str, config: RunnableConfig) -> str:
    """Useful for answering questions using the internal PDF knowledge base."""
    thread_id = config["configurable"].get("thread_id")
    retriever = pdf_retrievers.get(thread_id)

    if retriever is None:
        return "No PDF has been loaded yet. Please tell the user to upload a PDF in the sidebar first."

    try:
        response = retriever.invoke(query)
        if not response:
            return "No relevant information found in the PDF."
        retrieved_text = "\n\n---\n\n".join(doc.page_content for doc in response)
        return f"Retrieved Information:\n\n{retrieved_text}"
    except Exception as e:
        return f"An error occurred while retrieving information: {str(e)}"



@tool 
def purchase_stock(symbol:str,quantity:int) -> str:
    """Simulates a stock purchase. In a real-world scenario, this would interface with a brokerage API.NOTE: This is a mock function and does not perform real stock transactions."""
    # For demonstration purposes, we'll just return a confirmation message.
    decision = interrupt(f"Are you sure you want to purchase {quantity} shares of {symbol}? (yes/no)")
    if isinstance(decision, str) and decision.lower() == "yes":
        return f"Successfully purchased {quantity} shares of {symbol}."
    else:
        return f"Purchase of {quantity} shares of {symbol} was canceled."
    

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
GEMINI_MODEL = "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
tools = [calculator, get_stock_price, web_search,rag,purchase_stock]
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

