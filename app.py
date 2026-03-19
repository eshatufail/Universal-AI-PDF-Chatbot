import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ───────────────────────────────────────────────
#          Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI PDF Reader",
    page_icon="📚",
    layout="wide"
)

# ───────────────────────────────────────────────
#          Load secrets (API keys)
# ───────────────────────────────────────────────
# Recommended: Add these in Streamlit Cloud → Secrets
# or in .streamlit/secrets.toml locally
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, AttributeError):
    GROQ_API_KEY = None

# Optional: you can also support HUGGINGFACEHUB_API_TOKEN if you use paid/inference API
try:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except (KeyError, AttributeError):
    pass  # many open models work without token

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it in Streamlit Secrets.")
    st.stop()

# ───────────────────────────────────────────────
#          UI Styling
# ───────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background: #fdfbf7; }
    .stButton > button {
        background-color: #4a5d4e;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Georgia';
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Universal AI PDF Analyst")

# ───────────────────────────────────────────────
#          Sidebar – Upload & Settings
# ───────────────────────────────────────────────
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    analyze_clicked = st.button("Analyze PDF", use_container_width=True)

    st.markdown("---")
    st.caption("Using Groq + free HuggingFace embeddings")

# ───────────────────────────────────────────────
#          Process uploaded PDF
# ───────────────────────────────────────────────
if uploaded_file and analyze_clicked:
    with st.spinner("Processing document — may take 10–60 seconds..."):
        try:
            # Save uploaded file temporarily
            with open("temp_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 1. Load PDF
            loader = PyPDFLoader("temp_file.pdf")
            data = loader.load()

            # 2. Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=120,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            chunks = text_splitter.split_documents(data)

            # 3. Create embeddings (free & good quality model)
            @st.cache_resource(show_spinner=False)
            def get_embeddings():
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )

            embeddings = get_embeddings()

            # 4. Create / replace vector store
            if "vector_db" in st.session_state:
                del st.session_state.vector_db

            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="pdf_chat"
            )

            st.session_state.vector_db = vector_db
            st.session_state.retriever = vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            st.success("✅ PDF processed successfully! You can now ask questions.")
            os.remove("temp_file.pdf")  # cleanup

        except Exception as e:
            st.error(f"Error during document processing:\n{str(e)}")


# ───────────────────────────────────────────────
#          Chat Interface
# ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask anything about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # ─── LLM ───
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.25,
                    max_tokens=2048
                )

                # ─── Prompt ───
                template = """You are a helpful PDF analyst. Answer the question based only on the following context.
If you don't know the answer or the information is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer in a clear, concise and well-structured way:"""

                prompt_template = ChatPromptTemplate.from_template(template)

                # ─── Retriever ───
                retriever = st.session_state.get("retriever")

                if not retriever:
                    st.warning("Please upload and analyze a PDF first.")
                    st.stop()

                # Get relevant chunks
                docs = retriever.invoke(prompt)
                context_text = "\n\n".join(doc.page_content for doc in docs)

                # ─── Chain ───
                chain = (
                    {"context": lambda _: context_text, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

                response = chain.invoke(prompt)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Error while generating answer:\n{str(e)}")
