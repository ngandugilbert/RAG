# This is the UI for the RAG Chat App
import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import Chat


# Streamlit page configuration
st.set_page_config(
    page_title="Policy AI App",
    page_icon="🤖",
    layout="wide",
)

def display_messages():
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        if is_user:
            message(msg, is_user=is_user, key=str(i), avatar_style="identicon")  
        else:
            message(msg, is_user=is_user, key=str(i), avatar_style="bottts")  
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))
        st.session_state["user_input"] = ""

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Processing {file.name} ..."):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = Chat()

    with st.sidebar:
        st.header("Chat Policy App", divider='rainbow')
        st.file_uploader(
            "Select a document",
            type=["pdf"],
            key="file_uploader",
            on_change=read_and_save_file,
            accept_multiple_files=True,
        )
        st.session_state["ingestion_spinner"] = st.empty()

    display_messages()

    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.text_input("Message", key="user_input", on_change=process_input)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    page()
