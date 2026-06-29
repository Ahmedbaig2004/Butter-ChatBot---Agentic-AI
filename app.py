import streamlit as st
from backend import chatbot,initialize_dynamic_rag,set_pdf_retriever,get_threads,pdf_retrievers # This imports the compiled graph engine from backend.py
from langchain_core.messages import HumanMessage
import uuid
import backend 
import tempfile
from langgraph.types import Command

# 1. Page Configuration Setup
st.set_page_config(page_title="Chatbutter AI", page_icon="🤖", layout="centered")
st.title("🤖 Chatbutter Studio")
st.caption("A stateful AI Chatbot orchestrated by LangGraph & Groq")

def extract_text(content):
    """Normalize message content to a plain string, regardless of provider shape."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def get_pending_interrupt(config):
    """Returns the Interrupt object if the graph is genuinely paused, else None.
    Always re-derives truth from the actual graph state instead of trusting
    a manually-maintained session_state flag (which can go stale)."""
    state = chatbot.get_state(config)
    if not state.next:
        return None
    task = next((t for t in state.tasks if t.interrupts), None)
    if task is None:
        return None
    return task.interrupts[0]


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
            text = extract_text(msg.content)

            if text.strip():
                st.markdown(text)


# 5. Handle New User Inputs (This replaces the old "while True" and "input()")
def run_turn(input_data):
    """input_data is either {'messages': [...]} for a new message, 
    or a Command(resume=...) for resuming after interrupt approval."""
    with st.chat_message("assistant"):
        tool_status = st.empty()
        
        def stream_gen():
            try:
                stream = chatbot.stream(input_data, config, stream_mode="messages")
                used_tools = set()
                for chunk, metadata in stream:
                    if metadata.get("langgraph_node") == "chat_node":
                        if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                            for tc in chunk.tool_call_chunks:
                                if tc.get("name"):
                                    used_tools.add(tc['name'])
                                    tools_formatted = " + ".join(f"**{n}**" for n in used_tools)
                                    tool_status.info(f"Used tool: {tools_formatted}", icon="⚙️")
                        text = extract_text(getattr(chunk, 'content', None))
                        if text:
                            yield text
            except Exception as e:
                yield f"\n\n⚠️ Something went wrong: {str(e)}"
        
        st.write_stream(stream_gen())

    # After the stream finishes, just rerun — the next pass will re-check
    # get_pending_interrupt(config) fresh, so there's no flag to keep in sync.
    st.rerun()


# Always re-derive whether we're paused on an interrupt, fresh, from the graph itself.
pending = get_pending_interrupt(config)

# Normal chat input — only show if nothing is pending approval
if pending is None:
    if user_message := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(user_message)
        run_turn({"messages": [HumanMessage(content=user_message)]})

# If paused on an interrupt, show approval buttons instead of the normal chat input
else:
    st.warning(pending.value)  # e.g. "Are you sure you want to purchase 10 shares of AAPL?"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, proceed", use_container_width=True):
            run_turn(Command(resume="yes"))
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            run_turn(Command(resume="no"))

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