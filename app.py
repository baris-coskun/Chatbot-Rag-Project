"""
Gen Digital RAG Chatbot
Run: streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

CHROMA_DIR = Path(__file__).parent / "chroma_db"

SYSTEM_PROMPT = """You are a knowledgeable and helpful assistant for Gen Digital RAG Chatbot.

Answer questions based on the provided context from Gen Digital company materials.
If a question is outside the context, say so honestly, but try to provide general guidance.

Be specific, cite facts and figures from the context. If asked about products or brands, explain them clearly.
If asked about company culture, careers, or AI initiatives, share the relevant information.

Context from Gen Digital materials:
{context}
"""


def get_retriever():
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Set the GOOGLE_API_KEY environment variable!")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.3,
    )


def ask(question: str, retriever, llm, history: list[dict]) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = llm.invoke(messages)
    return response.content


def main():
    st.set_page_config(
        page_title="Gen Digital RAG Chatbot",
        page_icon="🤖",
        layout="centered",
    )

    st.title("Gen Digital RAG Chatbot")
    st.caption("Ask anything about Gen Digital: products, brands, AI initiatives, culture, careers...")

    if not CHROMA_DIR.exists():
        st.error("Run `python ingest.py` first to load the data!")
        st.stop()

    if "retriever" not in st.session_state:
        st.session_state.retriever = get_retriever()
        st.session_state.llm = get_llm()

    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer = ask(
                        prompt,
                        st.session_state.retriever,
                        st.session_state.llm,
                        st.session_state.history[:-1],
                    )
                st.write(answer)
                st.session_state.history.append({"role": "assistant", "content": answer})
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    st.error("Gemini API quota exceeded. Wait a moment and try again, or generate a new API key at https://aistudio.google.com/apikey")
                else:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
