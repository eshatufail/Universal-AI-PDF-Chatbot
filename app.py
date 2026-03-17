import streamlit as st
import os  # <--- Yeh line top par hona zaroori hai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Aesthetic UI Setup ---
st.set_page_config(page_title="Universal AI PDF Analyst", page_icon="📚", layout="wide")
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f0f2f6 0%, #e0eafc 100%); }
    h1 { color: #2c3e50; text-align: center; font-family: 'Segoe UI'; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Universal AI PDF Analyst")
st.caption("Upload any PDF and chat with it instantly!")

# --- Sidebar ---
with st.sidebar:
    st.header("Document Upload")
    pdf_file = st.file_uploader("Upload your PDF here", type=["pdf"])
    
    if pdf_file and st.button("Analyze Document"):
        with st.spinner("Processing your document..."):
            # Temporary path define karna
            temp_path = "temp_uploaded_file.pdf"
            
            # File ko write karna
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Check karna ke file create hui ya nahi
            if os.path.exists(temp_path):
                try:
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    splits = text_splitter.split_documents(docs)
                    
                    vectorstore = Chroma.from_documents(
                        splits, 
                        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    )
                    st.session_state.retriever = vectorstore.as_retriever()
                    st.success("Analysis Complete!")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
            else:
                st.error("Failed to save the file. Please try again.")

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Secure API Key handling
    llm = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile")
    
    template = """You are an intelligent AI Document Analyst. 
    Use the following context to answer. If not in context, use general knowledge.
    Mention if the tone is Technical, Emotional, or Educational.

    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(template)
    
    retriever = st.session_state.get("retriever")
    context = ""
    if retriever:
        context = "\n\n".join([d.page_content for d in retriever.invoke(prompt)])
    
    chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    with st.chat_message("assistant"):
        response = chain.invoke(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
