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

# --- Aesthetic UI Setup ---
st.set_page_config(page_title="Universal AI PDF Analyst", page_icon="📚", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    h1 { color: #1e3a8a; text-align: center; font-family: 'Helvetica'; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Universal AI PDF Analyst")
st.caption("Professional PDF Analysis with Sentiment Insights")

# --- API Keys ---
# Note: Google API Key is needed for stable cloud embeddings
# You can get one for free at: https://aistudio.google.com/
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyAi83gu799qgshzuGq2koZ_Jge74kGHzvE" # Get this for free from Google AI Studio

# --- Sidebar ---
with st.sidebar:
    st.header("Document Center")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if pdf_file and st.button("Initialize Document"):
        with st.spinner("Analyzing content..."):
            # Clean up old file if exists
            temp_path = "temp_vault.pdf"
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            try:
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                
                # Using Google Embeddings for Cloud Stability
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                st.session_state.retriever = vectorstore.as_retriever()
                st.success("Document Analysis Complete!")
            except Exception as e:
                st.error(f"System Error: {str(e)}")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask me about the uploaded file..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    
    template = """You are a professional Document Analyst. 
    Context: {context}
    
    Question: {question}
    
    Instructions: 
    1. If the answer is in the context, provide a detailed response.
    2. If not, use your general knowledge but mention it.
    3. Sentiment: End with a one-word tone analysis (e.g., Tone: Informative).
    
    Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(template)
    
    retriever = st.session_state.get("retriever")
    context_text = ""
    if retriever:
        docs = retriever.invoke(prompt)
        context_text = "\n\n".join([d.page_content for d in docs])
    
    chain = (
        {"context": lambda x: context_text, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    with st.chat_message("assistant"):
        try:
            full_response = chain.invoke(prompt)
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error("API Limit reached or Connection Lost. Please try again.")
