import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="AI PDF Reader", page_icon="📚", layout="wide")

# --- Keys ---
# Behtreen hal ye hai ke aap ye keys Streamlit Secrets mein dalein
GOOGLE_API_KEY = "AIzaSyAi83gu799qgshzuGq2koZ_Jge74kGHzvE" 
GROQ_API_KEY = "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"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and st.button("Analyze PDF"):
        with st.spinner("Processing document..."):
            # Temporary file save karna
            with open("temp_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                loader = PyPDFLoader("temp_file.pdf")
                data = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)
                
                # ✅ FIX: Updated model to 'text-embedding-004'
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004", 
                    google_api_key=GOOGLE_API_KEY
                )
                
                # Vector Database setup
                vector_db = Chroma.from_documents(chunks, embeddings)
                st.session_state.retriever = vector_db.as_retriever()
                st.success("Document Analyzed! Now you can ask questions.")
            except Exception as e:
                # Agar phir bhi 404 aaye to ye block error handle karega
                st.error(f"Technical Error: {e}")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the PDF..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    
    template = """Use the document context to answer.
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt_md = ChatPromptTemplate.from_template(template)
    retriever = st.session_state.get("retriever")
    context = ""
    
    if retriever:
        docs = retriever.invoke(prompt)
        context = "\n\n".join([d.page_content for d in docs])
    
    chain = ({"context": lambda x: context, "question": RunnablePassthrough()} | prompt_md | llm | StrOutputParser())
    
    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
