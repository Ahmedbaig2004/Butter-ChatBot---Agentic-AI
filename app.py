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

config = {
    "configurable": {"thread_id": st.session_state["current_thread_id"]},
    "metadata": {"thread_id": st.session_state["current_thread_id"]},
    "run_name": "chat_trace"
}
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
            # Check if this past AI message was a tool execution
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    st.info(f"Used tool: **{tool_call['name']}**", icon="⚙️")
            
            # Only render text if the AI message actually contains text
            if msg.content.strip():
                st.markdown(msg.content)


# 5. Handle New User Inputs (This replaces the old "while True" and "input()")
if user_message := st.chat_input("Type your message here..."):
    # Instantly render user's message in the web UI
    with st.chat_message("user"):
        st.markdown(user_message)

    # Render the assistant chat bubble
    with st.chat_message("assistant"):
        
        # Create an empty container to hold the tool UI above the typing text
        tool_status = st.empty()
        
        def langchain_stream_generator():
            stream = chatbot.stream(
                {"messages": [HumanMessage(content=user_message)]}, 
                config, 
                stream_mode="messages"
            )
            
            # Keep track of tools used in this exact run so they don't overwrite each other
            used_tools = set()
            
            for chunk, metadata in stream:
                if metadata.get("langgraph_node") == "chat_node":
                    
                    # Catch the AI deciding to call a tool
                    if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                        for tc in chunk.tool_call_chunks:
                            if tc.get("name"):
                                used_tools.add(tc['name'])
                                # Join multiple tools together if the AI uses more than one
                                tools_formatted = " + ".join([f"**{name}**" for name in used_tools])
                                tool_status.info(f"Used tool: {tools_formatted}", icon="⚙️")
                    
                    # Yield standard text for the typing animation
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
                
                # NOTE: We removed the green "Tool execution finished!" block here 
                # because it was overwriting the blue info box!

        # Stream the actual text below the status box
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
