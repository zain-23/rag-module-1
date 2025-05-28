from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss

loader = PyPDFLoader("data/attention.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splits = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()
vectorstore = faiss.FAISS.from_documents(splits, embeddings)
vectorstore.save_local("vectorStore")