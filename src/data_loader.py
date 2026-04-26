import pdfplumber
import json
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------
# 1. Extract text from PDF
# -------------------------------
def extract_text_from_pdf(pdf_path):
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    return full_text


# -------------------------------
# 2. Clean text
# -------------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d{4}-\d{2}', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,?%()\s]', '', text)

    return text.strip()


# -------------------------------
# 3. Chunk text
# -------------------------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


# -------------------------------
# 4. Process all PDFs
# -------------------------------
def process_all_pdfs(folder_path):
    all_chunks = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)

            print(f"📄 Processing {file}...")

            text = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(text)
            chunks = chunk_text(cleaned)

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": file.replace(".pdf", "")
                })

    return all_chunks


# -------------------------------
# 5. Save to JSONL
# -------------------------------
def save_to_jsonl(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f)
            f.write("\n")


# -------------------------------
# 6. Main
# -------------------------------
if __name__ == "__main__":
    folder_path = "data/raw"   # folder containing chapter PDFs
    output_path = "data/processed/class8_science.jsonl"

    print("📄 Processing all chapters...")

    chunks = process_all_pdfs(folder_path)

    print("💾 Saving...")
    save_to_jsonl(chunks, output_path)

    print(f"✅ Done! Total chunks created: {len(chunks)}")