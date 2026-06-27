import streamlit as st
from backend import chatbot  # This imports the compiled graph engine from backend.py
from langchain_core.messages import HumanMessage
import uuid
from backend import get_threads

# 1. Page Configuration Setup
st.set_page_config(page_title="Chatbutter AI", page_icon="🤖", layout="centered")
st.title("🤖 Chatbutter Studio")
st.caption("A stateful AI Chatbot orchestrated by LangGraph & Groq")



# 3. Pull historical messages directly from the LangGraph Checkpointer State


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_threads()
if "current_thread_id" not in st.session_state:
    if len(st.session_state["chat_threads"]) > 0:
        st.session_state["current_thread_id"] = st.session_state["chat_threads"][-1]

    else:
        first_id = str(uuid.uuid4())
        st.session_state["chat_threads"].append(first_id)
        st.session_state["current_thread_id"] = first_id

config = {"configurable": {"thread_id": st.session_state["current_thread_id"]}}
current_state = chatbot.get_state(config)
existing_messages = current_state.values.get("messages", [])

# 2. Thread Configuration for Memory
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
with st.sidebar:
    st.title("💬 Chat History")
    
    # "New Chat" Button
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state["chat_threads"].append(new_id)
        st.session_state["current_thread_id"] = new_id
        st.rerun() # Refresh the page to load the clean slate
    
    st.divider()
    
    # Render a list of clickable chat options
    for idx, t_id in enumerate(st.session_state["chat_threads"]):
        # Highlight the currently active chat thread
        is_current = (t_id == st.session_state["current_thread_id"])
        type_label = "👉 Active Chat" if is_current else f"Chat Session {idx + 1}"
        
        if st.button(type_label, key=t_id, use_container_width=True):
            st.session_state["current_thread_id"] = t_id
            st.rerun() # Refresh the page to load the clicked chat's history
