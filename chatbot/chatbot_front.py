# chatbot_front.py

import streamlit as st
from pathlib import Path
import os
from chatbot_text_extraction import extract_text, save_extracted_text
from chatbot_data_extraction import process_from_extracted_text
from chatbot_generate_chunks import process_file
from chatbot_generate_embeddings import process_chunked_data
from chatbot_faiss_index import process_embeddings_file
from chatbot_query_answering import answer_query_from_documents_debug
from chatbot_question_suggester import render_question_suggester
from streamlit_echarts import st_echarts

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

st.set_page_config(
    page_title="Crypto Fund Due Diligence Chatbot",
    page_icon="🪙",
    layout="wide"
)

# === STYLE ===
st.markdown("""
<style>
.stChatMessage { font-size: 1rem; }
.message-user {
    color: #0A9396; font-weight: bold; display: flex; align-items: center;
}
.message-assistant {
    color: #001219; font-weight: normal; display: flex; align-items: center;
}
.avatar {
    width: 24px; height: 24px; border-radius: 50%; margin-right: 0.5em;
}
.source-box {
    font-size: 0.85rem; background-color: #f0f0f0;
    padding: 0.5em; margin-top: 0.5em; border-radius: 5px;
}
.verdict-tag {
    display: inline-block; padding: 0.4em 0.8em;
    border-radius: 0.5em; color: white; font-size: 0.9rem;
    margin-top: 0.5rem;
}
.verdict-reliable { background-color: #4CAF50; }
.verdict-risky { background-color: #FF5722; }
</style>
""", unsafe_allow_html=True)

# === SESSION STATE ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === FOLDERS ===
UPLOAD_DIR = Path("upload")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# === HEADER ===
st.title("🪙 Crypto Fund Due Diligence Assistant")
st.markdown("Upload your documents and ask deep-dive questions about any crypto investment fund.")

# === FILE UPLOAD ===
uploaded_files = st.file_uploader("📁 Upload your files (PDF, DOCX, etc.)", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = UPLOAD_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Uploaded: {file.name}")

        progress_text = st.empty()
        progress_bar = st.progress(0)

        text = extract_text(file_path)
        progress_bar.progress(10)
        progress_text.text("Step 1/5: Extracting text...")

        if text and len(text.strip()) > 50:
            save_extracted_text(file_path, text)
            stem = file_path.stem.replace(" ", "_").replace(".", "_")
            text_path = OUTPUT_DIR / stem / "extracted_text.txt"

            progress_bar.progress(30)
            progress_text.text("Step 2/5: Extracting structured data...")
            process_from_extracted_text(text_path, stem)

            progress_bar.progress(50)
            progress_text.text("Step 3/5: Generating semantic chunks...")
            process_file(text_path, OUTPUT_DIR / stem / "extracted_data.json")

            progress_bar.progress(70)
            progress_text.text("Step 4/5: Generating embeddings...")
            process_chunked_data(OUTPUT_DIR / stem / "chunked_data.json")

            progress_bar.progress(90)
            progress_text.text("Step 5/5: Creating FAISS index...")
            process_embeddings_file(OUTPUT_DIR / stem / "chunked_data_with_embeddings.json")

            progress_bar.progress(100)
            progress_text.text("✅ Processing complete.")
        else:
            st.warning(f"Not enough extractable text in: {file.name}")

# === QUESTION SUGGESTER
st.markdown("---\n### 🧠 Suggested Due Diligence Questions")
render_question_suggester()

# === Q&A SECTION
st.markdown("---\n### 💬 Ask a question about your documents")
use_graph = st.toggle("🧠 Use GraphRAG instead of FAISS", value=False)

injected = st.session_state.get("injected_question", "")
question = st.text_input("Your question:", value=injected)

if st.button("Send") and question:
    with st.spinner("Searching and reviewing..."):
        result = answer_query_from_documents_debug(
            question,
            use_graph=use_graph
        )
        answer = result["answer"]
        sources = result["sources"]
        review = result["review"]
        missing = result["missing_points"]

        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", answer))

        st.markdown("🧠 <b>GraphRAG Activated</b>" if use_graph else "<i>Using FAISS index</i>", unsafe_allow_html=True)
        st.markdown(f"🔢 <b>Confidence Score:</b> {review['confidence_score']:.1f}%", unsafe_allow_html=True)
        st.progress(min(int(review["confidence_score"]), 100))

        with st.expander("📊 Visual Confidence Gauge"):
            st_echarts({
                "series": [{
                    "type": 'gauge',
                    "progress": {"show": True, "width": 18},
                    "axisLine": {"lineStyle": {"width": 18}},
                    "pointer": {"length": '80%', "width": 6},
                    "detail": {"valueAnimation": True, "formatter": '{value}%', "fontSize": 18},
                    "data": [{"value": review["confidence_score"]}]
                }]
            }, height=300)

        tag_class = "verdict-reliable" if review["hallucination_risk"] in ["none", "low"] else "verdict-risky"
        st.markdown(f"<div class='verdict-tag {tag_class}'>🧠 {review['verdict']}</div>", unsafe_allow_html=True)

        if review["justification"]:
            st.markdown(f"<p style='font-size:0.9rem;'>📝 <b>Justification:</b> {review['justification']}</p>", unsafe_allow_html=True)

        if sources:
            st.markdown("<div class='source-box'><b>Sources:</b><ul>" + "".join(f"<li>{src}</li>" for src in sources) + "</ul></div>", unsafe_allow_html=True)

        # Chunks
        with st.expander("📚 Chunks used in the answer"):
            for i, chunk in enumerate(review["context_chunks"]):
                if "chunk" in chunk:
                    meta = chunk["chunk"].get("metadata", {})
                    st.markdown(f"""
                        <div style='background:#f9f9f9; border:1px solid #ddd; padding:10px; margin-bottom:10px; border-radius:8px; font-size:0.9em;'>
                            <b>Chunk #{i+1}</b> — <code>{meta.get('chunk_id', 'N/A')}</code> <i>({meta.get('section', '')})</i><br>
                            {chunk["chunk"]["text"][:500]}...
                        </div>
                    """, unsafe_allow_html=True)

        # Missing points
        if missing:
            with st.expander("🔍 Missing Information for a Perfect Answer"):
                st.markdown("The following points are missing from the current context:")
                for mp in missing:
                    st.markdown(f"- ❓ {mp}")

# === CHAT HISTORY ===
for role, message in st.session_state.chat_history:
    avatar = "456/456212.png" if role == "user" else "4712/4712100.png"
    st.markdown(f"<div class='message-{role}'><img class='avatar' src='https://cdn-icons-png.flaticon.com/512/{avatar}'> {message}</div>", unsafe_allow_html=True)

# === RESET CHAT ===
if st.button("🧹 Clear chat history"):
    st.session_state.chat_history = []
    st.session_state["injected_question"] = ""
    st.experimental_rerun()
