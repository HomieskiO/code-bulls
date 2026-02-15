import os
import json
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURATION ---
JSON_FILE = "labeled_trading_quotes.json"
DB_DIR = "./chroma_db"


def build_persistent_brain():
    print("🏗️  Starting to build the local brain (No API Key needed)...")

    if not os.path.exists(JSON_FILE):
        print(f"❌ Error: {JSON_FILE} not found.")
        return

    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for item in data:
        page_content = f"Category: {item['category']} | Rule: {item['quote']}"
        metadata = {
            "category": item['category'],
            "impact": item['impact'],
            "source_url": item.get('source_url', ''),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"📚 Processing {len(documents)} trading quotes...")

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    # Use a high-quality local model (runs on your machine)
    # This is the 'Nano' equivalent for embeddings
    print("🧬 Loading local embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 Saving vector database to local disk...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print(f"✅ SUCCESS: Local brain built at: {DB_DIR}")


if __name__ == "__main__":
    build_persistent_brain()