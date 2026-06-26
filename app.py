import streamlit as st
from backend import chatbot  # This imports the compiled graph engine from backend.py
from langchain_core.messages import HumanMessage

# 1. Page Configuration Setup
st.set_page_config(page_title="Chatbutter AI", page_icon="🤖", layout="centered")
st.title("🤖 Chatbutter Studio")
st.caption("A stateful AI Chatbot orchestrated by LangGraph & Groq")

# 2. Thread Configuration for Memory
THREAD_ID = "streamlit_session_1"
config = {"configurable": {"thread_id": THREAD_ID}}

# 3. Pull historical messages directly from the LangGraph Checkpointer State
current_state = chatbot.get_state(config)
existing_messages = current_state.values.get("messages", [])

# 4. Display past conversation history visually on screen
for msg in existing_messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif msg.type == "ai":
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 5. Handle New User Inputs (This replaces the old "while True" and "input()")
if user_message := st.chat_input("Type your message here..."):
    # Instantly render user's message in the web UI
    with st.chat_message("user"):
        st.markdown(user_message)

    # Render the assistant chat bubble
    with st.chat_message("assistant"):
        
        def langchain_stream_generator():
            stream = chatbot.stream(
                {"messages": [HumanMessage(content=user_message)]}, 
                config, 
                stream_mode="messages"
            )
            
            for chunk, metadata in stream:
                if metadata.get("langgraph_node") == "chat_node":
                    if chunk.content:
                        yield chunk.content

        st.write_stream(langchain_stream_generator())