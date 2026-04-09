"""
Load data from data/gen_digital.txt, split into chunks, and store in ChromaDB with Gemini embeddings.
Run once: python ingest.py
Requires: GOOGLE_API_KEY environment variable (set via export or .env file)
"""

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

DATA_FILE = Path(__file__).parent / "data" / "gen_digital.txt"
CHROMA_DIR = Path(__file__).parent / "chroma_db"


def main():
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set the GOOGLE_API_KEY environment variable.")
        print("  export GOOGLE_API_KEY='your-key'")
        raise SystemExit(1)

    print(f"Loading data from {DATA_FILE} ...")
    text = DATA_FILE.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n============", "\n---", "\n\n", "\n", " "],
    )
    chunks = splitter.create_documents([text])
    print(f"Split into {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    print("Creating vector store in ChromaDB ...")
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Done! Stored {len(chunks)} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
