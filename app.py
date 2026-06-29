import streamlit as st
from backend import chatbot,initialize_dynamic_rag,set_pdf_retriever,get_threads,pdf_retrievers # This imports the compiled graph engine from backend.py
from langchain_core.messages import HumanMessage
import uuid
import backend 
import tempfile

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
            if msg.tool_calls:
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
                                before = len(used_tools)
                                used_tools.add(tc['name'])
                                if len(used_tools) > before:   # only re-render if something new was actually added
                                    tools_formatted = " + ".join([f"**{name}**" for name in used_tools])
                                    tool_status.info(f"Used tool: {tools_formatted}", icon="⚙️")
                    
                    # Yield standard text for the typing animation
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
                


        # Stream the actual text below the status box
        st.write_stream(langchain_stream_generator())
with st.sidebar:
    st.title("💬 Chat History")
    
    # "New Chat" Button
    if st.button("➕ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state["chat_threads"].append(new_id)
        st.session_state["current_thread_id"] = new_id
        st.rerun()
    
    st.divider()
    
    # Render a list of clickable chat options
    for idx, t_id in enumerate(st.session_state["chat_threads"]):
        is_current = (t_id == st.session_state["current_thread_id"])
        type_label = "👉 Active Chat" if is_current else f"Chat Session {idx + 1}"
        
        if st.button(type_label, key=t_id, use_container_width=True):
            st.session_state["current_thread_id"] = t_id
            st.rerun()
            
    st.divider()
    st.subheader("📁 Upload Knowledge Base")


    current_thread = st.session_state["current_thread_id"]
    has_pdf_for_thread = current_thread in pdf_retrievers

    if has_pdf_for_thread:
        st.success("📄 This chat has a PDF loaded.")
    else:
        st.info("No PDF attached to this chat yet.")

    uploaded_file = st.file_uploader("Upload a PDF to chat with it", type=["pdf"])

    if uploaded_file is not None:
        already_done = st.session_state.get("processed_file_per_thread", {}).get(current_thread) == uploaded_file.name

        if not already_done:
            with st.spinner("Processing PDF and building vector index..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                retriever = initialize_dynamic_rag(temp_file_path)

                set_pdf_retriever(current_thread, retriever)
                st.session_state.setdefault("processed_file_per_thread", {})[current_thread] = uploaded_file.name

                st.success(f"Successfully indexed: {uploaded_file.name}!")
