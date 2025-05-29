from langchain_community.vectorstores import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import  load_dotenv
from langchain.prompts import PromptTemplate
import os

load_dotenv()

db = faiss.FAISS.load_local("vectorStore", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
retriever = db.as_retriever()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
                                               
Context: {context}
                                               
Question: {question}
""")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
)

def ask_question(query: str):
    """
    Ask a question to the RAG system and get an answer along with source documents.
    
    Args:
        question (str): The question to ask.
        
    Returns:
        dict: A dictionary containing the answer and source documents.
    """
    result = qa(query)
    return result["result"]