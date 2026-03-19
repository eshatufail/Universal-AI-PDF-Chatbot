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
st.set_page_config(
    page_title="AI PDF Analyst",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Professional clean look ─────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .st-emotion-cache-1v0mbdj {font-family: 'Inter', sans-serif !important;}
        section[data-testid="stSidebar"] > div:first-child {font-family: 'Inter', sans-serif !important;}
        .stChatMessage {padding: 1rem !important;}
        .stChatInput > div:first-child {font-family: 'Inter', sans-serif;}
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Secrets ─────────────────────────────────────────────
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing → please set it in .streamlit/secrets.toml")
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
            messages TEXT, -- JSON string
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

# ─── Session state init ───────────────────────────────────
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
            with st.form("login_form", clear_on_submit=False):
                login_user = st.text_input("Username", key="login_user")
                login_pass = st.text_input("Password", type="password", key="login_pass")
                if st.form_submit_button("Login", use_container_width=True):
                    if authenticate(login_user, login_pass):
                        st.session_state.authenticated = True
                        st.session_state.username = login_user
                        st.session_state.current_messages = []
                        st.success("Logged in successfully")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        with tab2:
            with st.form("register_form", clear_on_submit=True):
                reg_user = st.text_input("Choose username", key="reg_user")
                reg_pass = st.text_input("Choose password", type="password", key="reg_pass")
                if st.form_submit_button("Register", use_container_width=True):
                    if register_user(reg_user, reg_pass):
                        st.success("Account created. Please log in.")
                    else:
                        st.error("Username already taken")

    else:
        st.header(f"👤 {st.session_state.username}")

        if st.button("Logout", use_container_width=True):
            if st.session_state.current_messages and len(st.session_state.current_messages) > 2:
                title = (st.session_state.current_messages[0]["content"][:40] + "...").strip()
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
        st.subheader("Previous Conversations")
        convs = load_user_conversations(st.session_state.username)
        for conv in convs:
            label = f"{conv['title']}  ({conv['date'][:10]})"
            if st.button(label, key=f"load_{conv['id']}", use_container_width=True):
                loaded = get_conversation_by_id(conv['id'], st.session_state.username)
                if loaded:
                    st.session_state.current_messages = loaded
                    st.rerun()

        st.markdown("---")
        with st.container():
            st.header("Document")
            uploaded_file = st.file_uploader("Upload PDF", type="pdf", help="Only PDF files are supported")
            analyze = st.button("Analyze PDF", use_container_width=True, type="primary")

# ─── Main content ────────────────────────────────────────
if not st.session_state.authenticated:
    st.title("📚 AI PDF Analyst")
    st.info("Please login or register using the sidebar to continue.")
    st.stop()

# ─── Authenticated layout ────────────────────────────────
main_col, info_col = st.columns([7, 3])

with main_col:
    st.title("📚 AI PDF Analyst")
    st.caption("Ask precise questions about your document • Get accurate, sourced answers")

    # Chat history
    AVATAR_USER = "👤"
    AVATAR_AI   = "🤖"

    for msg in st.session_state.current_messages:
        avatar = AVATAR_USER if msg["role"] == "user" else AVATAR_AI
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.current_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AVATAR_AI):
            with st.status("Processing your question...", expanded=True) as status:
                st.write("Searching relevant passages...")
                try:
                    llm = ChatGroq(
                        groq_api_key=GROQ_API_KEY,
                        model_name="llama-3.3-70b-versatile",
                        temperature=0.3
                    )

                    template = """Answer using only this context. If unsure or the information is not in the document → say so clearly.

Context:
{context}

Question: {question}

Clear, concise and accurate answer:"""

                    prompt_tpl = ChatPromptTemplate.from_template(template)

                    retriever = st.session_state.get("retriever")
                    if not retriever:
                        st.warning("Please upload and analyze a PDF first.")
                        status.update(label="No document loaded", state="error", expanded=False)
                    else:
                        docs = retriever.invoke(prompt)
                        ctx = "\n\n".join(d.page_content for d in docs)

                        chain = (
                            {"context": lambda _: ctx, "question": RunnablePassthrough()}
                            | prompt_tpl
                            | llm
                            | StrOutputParser()
                        )

                        st.write("Generating answer...")
                        response = chain.invoke(prompt)

                        st.markdown(response)
                        st.session_state.current_messages.append({"role": "assistant", "content": response})

                        status.update(label="Answer complete", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    status.update(label="Error occurred", state="error", expanded=False)

# Right column - document info
with info_col:
    st.subheader("Document Status")
    if "vectorstore" in st.session_state:
        try:
            count = len(st.session_state.vectorstore.get()["ids"])
            st.success(f"Document loaded\n**{count:,}** text chunks indexed")
        except:
            st.warning("Document index loaded but cannot read statistics")
    else:
        st.info("Upload a PDF and click **Analyze PDF** to start asking questions.")

    st.markdown("**Best results when you**")
    st.markdown("• Ask one clear question at a time")
    st.markdown("• Be specific about sections or topics")
    st.markdown("• Upload focused, well-structured documents")

# Auto-save every ~6 messages
if len(st.session_state.current_messages) % 6 == 0 and len(st.session_state.current_messages) > 4:
    if st.session_state.current_messages:
        title = (st.session_state.current_messages[0]["content"][:35] + "...").strip()
        save_conversation(
            st.session_state.username,
            title,
            st.session_state.current_messages
        )
