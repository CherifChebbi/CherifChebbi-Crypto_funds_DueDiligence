# chatbot_front.py

import streamlit as st
from pathlib import Path
import os
from chatbot_text_extraction import extract_text, save_extracted_text
from chatbot_data_extraction import process_from_extracted_text
from chatbot_generate_chunks import process_file
from chatbot_generate_embeddings import process_chunked_data
from chatbot_faiss_index import process_embeddings_file
from chatbot_query_answering import answer_query_from_documents

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Crypto Fund Due Diligence Chatbot",
    page_icon="🧠",
    layout="wide"
)

# === STYLE ===
st.markdown("""
    <style>
        .stChatMessage { font-size: 1rem; }
        .message-user {
            color: #0A9396;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        .message-assistant {
            color: #001219;
            font-weight: normal;
            display: flex;
            align-items: center;
        }
        .avatar {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            margin-right: 0.5em;
        }
        .source-box {
            font-size: 0.85rem;
            background-color: #f0f0f0;
            padding: 0.5em;
            margin-top: 0.5em;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# === SESSION STATE ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# === FOLDERS ===
UPLOAD_DIR = Path("upload")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# === HEADER ===
st.title("🪙 Crypto Fund Due Diligence Assistant")
st.markdown("Ask questions about any uploaded documents related to crypto investment funds.")

# === FILE UPLOAD ===
uploaded_files = st.file_uploader("📁 Upload your files (PDF, DOCX, etc.)", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = UPLOAD_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Uploaded: {file.name}")

        text = extract_text(file_path)
        if text and len(text.strip()) > 50:
            save_extracted_text(file_path, text)
            stem = file_path.stem.replace(" ", "_").replace(".", "_")
            text_path = OUTPUT_DIR / stem / "extracted_text.txt"
            process_from_extracted_text(text_path, stem)
            process_file(text_path, OUTPUT_DIR / stem / "extracted_data.json")
            process_chunked_data(OUTPUT_DIR / stem / "chunked_data.json")
            process_embeddings_file(OUTPUT_DIR / stem / "chunked_data_with_embeddings.json")
        else:
            st.warning(f"Not enough extractable text in: {file.name}")

st.markdown("---\n## 💬 Ask a question about your documents")

# === CHAT ===
question = st.text_input("Your question:")
if st.button("Send") and question:
    with st.spinner("Searching..."):
        answer, sources = answer_query_from_documents(question)
        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", answer))
        st.session_state.last_sources = sources

# === CHAT HISTORY DISPLAY ===
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"<div class='message-user'><img class='avatar' src='https://cdn-icons-png.flaticon.com/512/456/456212.png'> {message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='message-assistant'><img class='avatar' src='https://cdn-icons-png.flaticon.com/512/4712/4712100.png'> {message}</div>",
            unsafe_allow_html=True
        )

# === SOURCES (from last assistant response)
if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "assistant":
    if st.session_state.last_sources:
        st.markdown("<div class='source-box'><b>Sources:</b><ul>" +
                    "".join(f"<li>{src}</li>" for src in st.session_state.last_sources) +
                    "</ul></div>", unsafe_allow_html=True)

# === RESET BUTTON ===
if st.button("🧹 Clear chat history"):
    st.session_state.chat_history = []
    st.session_state.last_sources = []
    st.experimental_rerun()
