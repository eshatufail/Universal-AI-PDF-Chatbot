import streamlit as st
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
st.caption("Upload any PDF (Novel, Study Notes, Tech Docs) and start chatting!")

# --- Sidebar ---
with st.sidebar:
    st.header("Document Upload")
    pdf_file = st.file_uploader("Upload your PDF here", type=["pdf"])
    if pdf_file and st.button("Analyze Document"):
        with st.spinner("Processing your document..."):
            with open("temp.pdf", "wb") as f: f.write(pdf_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            # Sahi tarah se split karna
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            
            vectorstore = Chroma.from_documents(splits, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
            st.session_state.retriever = vectorstore.as_retriever()
            st.success("Document analyzed! You can ask anything.")

# --- Chat Logic ---
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask about the document or general concepts..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Apni API Key yahan safe rakhein ya Streamlit secrets use karein
    llm = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile")
    
    template = """You are an intelligent AI Document Analyst. 
    1. If the user asks about the uploaded document, provide an answer based strictly on the provided context.
    2. If the user asks a general conceptual question not found in the document, use your vast general knowledge to explain it clearly.
    3. Sentiment/Tone Analysis: Also briefly mention if the content discussed is 'Technical', 'Emotional', 'Educational', or 'Complex'.

    Context (from PDF): {context}
    
    Question: {question}
    Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(template)
    
    # Retrieval
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