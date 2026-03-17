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

# --- UI Setup ---
st.set_page_config(page_title="Universal AI PDF Analyst", page_icon="📚", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    h1 { color: #1e3a8a; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Universal AI PDF Analyst")

# --- Keys ---
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyAi83gu799qgshzuGq2koZ_Jge74kGHzvE" # Apni key yahan lagayein

# Connection stability ke liye environment variable set karna
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Center")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if pdf_file and st.button("Initialize Document"):
        with st.spinner("Analyzing..."):
            temp_path = "temp_vault.pdf"
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
                
                # TASK-READY Embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=GOOGLE_API_KEY,
                    task_type="retrieval_queries" # Stability ke liye add kiya
                )
                
                # Database setup
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.session_state.retriever = vectorstore.as_retriever()
                st.success("Document Loaded Successfully!")
            except Exception as e:
                st.error(f"Connection Issue: {str(e)}")

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    
    template = """You are an AI Document Analyst. Answer based on context:
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(template)
    retriever = st.session_state.get("retriever")
    context_text = ""
    
    if retriever:
        try:
            docs = retriever.invoke(prompt)
            context_text = "\n\n".join([d.page_content for d in docs])
        except:
            context_text = "Could not fetch context from PDF."

    chain = ({"context": lambda x: context_text, "question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())
    
    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
