import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Change here
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- API Keys ---
# Behtar hai ke aap inhein Streamlit Secrets mein rakhein
GROQ_KEY = "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"
# Google Embeddings ke liye bhi Groq jaisi aik free API key chahiye hoti hai
# Agar aapke paas nahi hai to aap purana HuggingFace wala code hi use karein
# lekin niche diye gaye 'try-except' block ke sath.

# --- Sidebar Logic with Extra Stability ---
with st.sidebar:
    st.header("Document Upload")
    pdf_file = st.file_uploader("Upload your PDF here", type=["pdf"])
    
    if pdf_file and st.button("Analyze Document"):
        with st.spinner("Processing..."):
            temp_path = "temp_file.pdf"
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            if os.path.exists(temp_path):
                try:
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
                    
                    # HuggingFace ko 'cpu' device par force karna taake error na aaye
                    embeddings = HuggingFaceEmbeddings(
                        model_name="all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'} 
                    )
                    
                    vectorstore = Chroma.from_documents(splits, embeddings)
                    st.session_state.retriever = vectorstore.as_retriever()
                    st.success("Ready!")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
