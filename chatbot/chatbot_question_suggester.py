# chatbot_question_suggester.py

import streamlit as st
import json
from pathlib import Path

QUESTION_BANK_PATH = Path("question_bank/question_bank_structured.json")

@st.cache_data
def load_question_bank():
    with open(QUESTION_BANK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def render_question_suggester():
    st.markdown("Browse categorized questions to guide your crypto fund investigation:")

    question_bank = load_question_bank()
    categories = sorted(question_bank.keys())

    selected_category = st.selectbox("📂 Select a category", categories)

    if selected_category:
        questions = question_bank[selected_category]
        for q in questions:
            if st.button(f"❓ {q['question']}", key=q['question']):
                st.session_state["injected_question"] = q["question"]
