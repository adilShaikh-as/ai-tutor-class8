import streamlit as st
import json
import re
import os
from langchain_ollama import ChatOllama

# 🔇 Fix transformer warnings
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Tutor", layout="centered")

st.title("📘 Class 8 Science AI Tutor")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    texts = []
    sources = []
    with open("data/processed/class8_science.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            sources.append(data["source"])
    return texts, sources

texts, sources = load_data()

# -------------------- SEARCH --------------------
def clean_words(text):
    return re.findall(r'\b\w+\b', text.lower())

def improved_search(query, texts, top_k=3):
    query_words = set(clean_words(query))

    # Important keywords boost
    keywords = ["photosynthesis", "plant", "chlorophyll", "sunlight"]

    scores = []
    for i, text in enumerate(texts):
        text_lower = text.lower()
        text_words = set(clean_words(text_lower))

        # base score
        score = len(query_words.intersection(text_words))

        # keyword boost
        score += sum(2 for k in keywords if k in text_lower)

        # phrase match boost (VERY IMPORTANT)
        if query.lower() in text_lower:
            score += 5

        scores.append((score, i))

    scores.sort(reverse=True)

    return [idx for score, idx in scores if score > 0][:top_k]

# -------------------- LLM --------------------
llm = ChatOllama(model="llama3")

def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "I couldn’t find this in Class 8 Science. Try rephrasing your question."

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a Class 8 Science tutor.

Answer ONLY from the given context.
Use simple language for students.

If the answer is not in the context, say:
"I’m focused on Class 8 Science."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content

# -------------------- SESSION --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- INPUT --------------------
if prompt := st.chat_input("Ask your question..."):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):

            try:
                indices = improved_search(prompt, texts)
                retrieved_chunks = [texts[i] for i in indices]

                answer = generate_answer(prompt, retrieved_chunks)

            except Exception as e:
                answer = "⚠️ Something went wrong. Please try again."

            st.markdown(answer)

            # Show sources
            if indices:
                with st.expander("📖 Sources"):
                    for i in indices:
                        st.write(f"📄 {sources[i]}")

    # Save AI response
    st.session_state.messages.append({"role": "assistant", "content": answer})