# 📘 Class 8 Science AI Tutor

A Retrieval-Augmented Generation (RAG) based AI tutor for NCERT Class 8 Science.  
It retrieves relevant textbook chunks and generates answers using a local LLM (Llama3 via Ollama).

---

## 🚀 Features
- 💬 ChatGPT-style interface (Streamlit)
- 📚 Answers grounded in NCERT Class 8 Science
- 🔎 Retrieval + generation (RAG)
- 📖 Source snippets for transparency
- 📴 Runs locally (no API cost)

---

## 🧠 How it works
1. User asks a question
2. System retrieves relevant chunks from the textbook (keyword-enhanced search)
3. Retrieved context is passed to Llama3 (Ollama)
4. Model generates a simple, grade-appropriate answer


---

## 🧰 Tech Stack
- Python
- Streamlit
- Ollama (Llama3)
- JSONL dataset

---

## ▶️ Run locally

```bash
# 1) install deps
pip install -r requirements.txt

# 2) run app
python -m streamlit run app/app.py
