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
from streamlit_echarts import st_echarts  # ✅ jauge circulaire

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

st.set_page_config(
    page_title="Crypto Fund Due Diligence Chatbot",
    page_icon="🪙",
    layout="wide"
)

# === STYLES ===
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
    .confidence-score {
        font-size: 0.9rem;
        color: #5A5A5A;
        margin-bottom: 1rem;
    }
    .verdict-tag {
        display: inline-block;
        padding: 0.4em 0.8em;
        border-radius: 0.5em;
        color: white;
        font-size: 0.9rem;
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
st.markdown("Ask questions about any uploaded documents related to crypto investment funds.")

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

# === Q&A SECTION ===
st.markdown("---\n### 💬 Ask a question about your documents")

use_web = st.toggle("📡 Use Web Sources", value=True)
use_graph = st.toggle("🧠 Use GraphRAG instead of FAISS", value=False)

question = st.text_input("Your question:")
if st.button("Send") and question:
    with st.spinner("Searching and reviewing..."):
        result = answer_query_from_documents_debug(
            question,
            use_web=use_web,
            use_graph=use_graph
        )
        answer = result["answer"]
        sources = result["sources"]
        review = result["review"]

        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("assistant", answer))

        # === REVIEW ===
        if use_graph:
            st.markdown("🧠 <b>GraphRAG Activated</b>", unsafe_allow_html=True)
        else:
            st.markdown("<i>Using FAISS index for document retrieval</i>", unsafe_allow_html=True)

        score = review["confidence_score"]
        st.markdown(f"🔢 <b>Confidence Score:</b> {score:.1f}%", unsafe_allow_html=True)
        st.progress(min(int(score), 100))

        with st.expander("📊 Visual Confidence Gauge"):
            st_echarts({
                "series": [{
                    "type": 'gauge',
                    "progress": {"show": True, "width": 18},
                    "axisLine": {"lineStyle": {"width": 18}},
                    "axisTick": {"show": False},
                    "splitLine": {"length": 15, "lineStyle": {"width": 2, "color": "#999"}},
                    "axisLabel": {"distance": 25, "color": "#999", "fontSize": 12},
                    "anchor": {"show": True, "showAbove": True, "size": 10, "itemStyle": {"color": "#999"}},
                    "pointer": {"icon": 'rect', "length": '80%', "width": 6, "itemStyle": {"color": 'auto'}},
                    "detail": {"valueAnimation": True, "formatter": '{value}%', "color": 'auto', "fontSize": 20},
                    "data": [{"value": score}]
                }]
            }, height=300)

        tag_class = "verdict-reliable" if review["hallucination_risk"] in ["none", "low"] else "verdict-risky"
        st.markdown(f"<div class='verdict-tag {tag_class}'>🧠 {review['verdict']}</div>", unsafe_allow_html=True)

        if review['justification']:
            st.markdown(f"<p style='font-size:0.9rem; color:#555;'>📝 <b>Justification:</b> {review['justification']}</p>", unsafe_allow_html=True)

        if sources:
            st.markdown("<div class='source-box'><b>Sources:</b><ul>" + "".join(f"<li>{src}</li>" for src in sources) + "</ul></div>", unsafe_allow_html=True)

        # === CHUNKS UTILISÉS ===
        with st.expander("📚 Chunks used in the answer"):
            graph_chunks = [c for c in review.get("context_chunks", []) if c.get("origin") == "graph"]
            st.markdown(f"🧠 <b>Chunks from GraphRAG:</b> {len(graph_chunks)}", unsafe_allow_html=True)
            for i, chunk in enumerate(review.get("context_chunks", [])):
                if "chunk" in chunk:
                    meta = chunk["chunk"].get("metadata", {})
                    source_id = meta.get("chunk_id", "N/A")
                    section = meta.get("section", "")
                    st.markdown(f"""
                        <div style='background:#f9f9f9; border:1px solid #ddd; padding:10px; margin-bottom:10px; border-radius:8px; font-size:0.9em;'>
                            <b>Chunk #{i+1}</b> — <code>{source_id}</code> <i>({section})</i><br>
                            {chunk['chunk']['text'][:500]}...
                        </div>
                    """, unsafe_allow_html=True)

# === CHAT HISTORY ===
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='message-user'><img class='avatar' src='https://cdn-icons-png.flaticon.com/512/456/456212.png'> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='message-assistant'><img class='avatar' src='https://cdn-icons-png.flaticon.com/512/4712/4712100.png'> {message}</div>", unsafe_allow_html=True)

# === RESET CHAT ===
if st.button("🧹 Clear chat history"):
    st.session_state.chat_history = []
    st.experimental_rerun()
