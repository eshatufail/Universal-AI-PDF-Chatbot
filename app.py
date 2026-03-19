import streamlit as st
import os
import sqlite3
import json
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ─── Page config ────────────────────────────────────────
st.set_page_config(page_title="AI PDF Analyst", page_icon="📚", layout="wide")

# ─── Secrets ─────────────────────────────────────────────
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing → please set it in secrets")
    st.stop()

# ─── SQLite setup ────────────────────────────────────────
DB_FILE = "users_chats.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            title TEXT,
            created_at TEXT,
            messages TEXT,           -- JSON string
            FOREIGN KEY(username) REFERENCES users(username)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ─── Auth helpers ────────────────────────────────────────
def authenticate(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None

def register_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def save_conversation(username, title, messages):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    messages_json = json.dumps(messages)
    c.execute("""
        INSERT INTO conversations (username, title, created_at, messages)
        VALUES (?, ?, ?, ?)
    """, (username, title, timestamp, messages_json))
    conn.commit()
    conn.close()

def load_user_conversations(username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations WHERE username = ? ORDER BY created_at DESC", (username,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "date": r[2]} for r in rows]

def get_conversation_by_id(conv_id, username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT messages FROM conversations WHERE id = ? AND username = ?", (conv_id, username))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

# ─── Auth UI / Routing ───────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.current_messages = []

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    if not st.session_state.authenticated:
        st.header("Login / Register")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form("login_form"):
                login_user = st.text_input("Username", key="login_user")
                login_pass = st.text_input("Password", type="password", key="login_pass")
                if st.form_submit_button("Login"):
                    if authenticate(login_user, login_pass):
                        st.session_state.authenticated = True
                        st.session_state.username = login_user
                        st.session_state.current_messages = []
                        st.success("Logged in!")
                        st.rerun()
                    else:
                        st.error("Wrong credentials")

        with tab2:
            with st.form("register_form"):
                reg_user = st.text_input("Choose username", key="reg_user")
                reg_pass = st.text_input("Choose password", type="password", key="reg_pass")
                if st.form_submit_button("Register"):
                    if register_user(reg_user, reg_pass):
                        st.success("Account created! Please login.")
                    else:
                        st.error("Username already taken")

    else:
        st.header(f"👤 {st.session_state.username}")
        
        if st.button("Logout"):
            # Optional: save current chat before logout
            if st.session_state.current_messages and len(st.session_state.current_messages) > 2:
                title = st.session_state.current_messages[0]["content"][:40] + "..."
                save_conversation(
                    st.session_state.username,
                    title,
                    st.session_state.current_messages
                )
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.current_messages = []
            st.rerun()

        st.markdown("---")
        st.subheader("Previous Chats")

        convs = load_user_conversations(st.session_state.username)
        for conv in convs:
            label = f"{conv['title']}  ({conv['date'][:10]})"
            if st.button(label, key=f"load_{conv['id']}"):
                loaded = get_conversation_by_id(conv['id'], st.session_state.username)
                if loaded:
                    st.session_state.current_messages = loaded
                    st.rerun()

# ─── Main content ────────────────────────────────────────
if not st.session_state.authenticated:
    st.title("📚 Universal AI PDF Analyst")
    st.info("Please login or register in the sidebar to continue.")
    st.stop()

# ─── Now user is authenticated ───────────────────────────

st.title("📚 Universal AI PDF Analyst")

# ─── File upload & processing ────────────────────────────
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    analyze = st.button("Analyze PDF", use_container_width=True)

if uploaded_file and analyze:
    with st.spinner("Processing PDF..."):
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900, chunk_overlap=120
            )
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            if "vectorstore" in st.session_state:
                del st.session_state.vectorstore

            st.session_state.vectorstore = Chroma.from_documents(
                chunks, embeddings, collection_name=f"pdf_{st.session_state.username}"
            )
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(k=4)

            st.success("PDF ready!")
            os.remove("temp.pdf")
        except Exception as e:
            st.error(f"Processing failed → {e}")

# ─── Chat ────────────────────────────────────────────────
if "current_messages" not in st.session_state:
    st.session_state.current_messages = []

for msg in st.session_state.current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.session_state.current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3
                )

                template = """Answer using only this context. If unsure → say so.

Context:
{context}

Question: {question}

Clear & concise answer:"""

                prompt_tpl = ChatPromptTemplate.from_template(template)

                retriever = st.session_state.get("retriever")
                if not retriever:
                    st.warning("Please upload & analyze a PDF first.")
                else:
                    docs = retriever.invoke(prompt)
                    ctx = "\n\n".join(d.page_content for d in docs)

                    chain = (
                        {"context": lambda _: ctx, "question": RunnablePassthrough()}
                        | prompt_tpl
                        | llm
                        | StrOutputParser()
                    )

                    response = chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.current_messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Error: {e}")

# Auto-save current chat every few messages (optional improvement)
if len(st.session_state.current_messages) % 6 == 0 and len(st.session_state.current_messages) > 4:
    title = st.session_state.current_messages[0]["content"][:35] + "..."
    save_conversation(
        st.session_state.username,
        title,
        st.session_state.current_messages
    )
