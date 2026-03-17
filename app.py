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

# --- UI Styling ---
st.markdown("""
    <style>
    .stApp { background: #fdfbf7; }
    .stButton>button { background-color: #4a5d4e; color: white; border-radius: 10px; width: 100%; }
    h1 { color: #2c3e50; font-family: 'Georgia'; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Universal AI PDF Analyst")

# --- API Keys ---
GOOGLE_API_KEY = "AIzaSyAi83gu799qgshzuGq2koZ_Jge74kGHzvE" 
GROQ_API_KEY = "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"

# Environment Variables (Very Important)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and st.button("Analyze PDF"):
        with st.spinner("Processing..."):
            with open("temp_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                loader = PyPDFLoader("temp_file.pdf")
                data = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)
                
                # SAFETY FIX: Embeddings setup with explicit key
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=GOOGLE_API_KEY,
                    task_type="retrieval_document"
                )
                
                # Vector Database
                if "retriever" in st.session_state:
                    del st.session_state.retriever # Purana data saaf karna

                vector_db = Chroma.from_documents(chunks, embeddings)
                st.session_state.retriever = vector_db.as_retriever()
                st.success("Ready to Chat!")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the PDF..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # LLM via Groq (Llama 3 is very fast)
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    
    template = """You are a helpful assistant. Use the context to answer. 
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt_md = ChatPromptTemplate.from_template(template)
    
    retriever = st.session_state.get("retriever")
    context_data = ""
    if retriever:
        try:
            # Context fetch karne mein agar block ho to skip karein
            docs = retriever.invoke(prompt)
            context_data = "\n\n".join([d.page_content for d in docs])
        except:
            context_data = "Context not available due to safety filters."
    
    chain = ({"context": lambda x: context_data, "question": RunnablePassthrough()} | prompt_md | llm | StrOutputParser())
    
    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
