import streamlit as st
from backend.rag import ask_question

st.title("🧠 Ask My Docs - RAG Assistant")

query = st.text_input("Ask a question about your documents:")

if query:
    result = ask_question(query)
    st.write(result)