"""
Enhanced RAG ChatBot using LangChain with LCEL patterns.
Modernized version with improved error handling, logging, and configuration.
"""

import streamlit as st
import os
import tempfile
import logging
from pathlib import Path
from typing import List, Optional
from streamlit_chat import message
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Import configuration
try:
    from config import settings
except ImportError:
    st.error("Configuration file not found. Please ensure config.py exists.")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def local_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
        <style>
            .big-font {
                font-size: 20px !important;
                color: #3498db;
                font-weight: 600;
            }
            .info-text {
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stButton>button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 24px;
                font-weight: 600;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
        </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "uploaded_texts" not in st.session_state:
        st.session_state["uploaded_texts"] = []
    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def get_llm():
    """
    Get the appropriate LLM based on configuration.
    Returns ChatOpenAI or ChatGroq instance.
    """
    try:
        provider = settings.get_llm_provider()
        
        if provider == "groq":
            if not settings.groq_api_key:
                st.error("Groq API key not found. Please set GROQ_API_KEY in .env file.")
                st.stop()
            logger.info(f"Using Groq with model: {settings.default_model}")
            return ChatGroq(
                model=settings.default_model,
                temperature=settings.model_temperature,
                groq_api_key=settings.groq_api_key,
                streaming=settings.enable_streaming
            )
        else:
            if not settings.openai_api_key:
                st.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
                st.stop()
            logger.info(f"Using OpenAI with model: {settings.default_model}")
            return ChatOpenAI(
                model=settings.default_model,
                temperature=settings.model_temperature,
                openai_api_key=settings.openai_api_key,
                streaming=settings.enable_streaming
            )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        st.error(f"Failed to initialize language model: {e}")
        st.stop()


def upload_and_process_files():
    """Handle file uploads and process documents."""
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "doc"],
        key="file_uploader",
        help=f"Maximum file size: {settings.max_upload_size_mb}MB"
    )
    
    if uploaded_files:
        new_files = False
        for file in uploaded_files:
            # Check if file already processed
            if file.name not in [doc.metadata.get("source", "") for doc in st.session_state["uploaded_texts"]]:
                try:
                    file_extension = os.path.splitext(file.name)[1].lower()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                        temp_file.write(file.read())
                        temp_file_path = temp_file.name
                    
                    # Load document based on file type
                    if file_extension == ".pdf":
                        loader = PyPDFLoader(temp_file_path)
                    elif file_extension == ".txt":
                        loader = TextLoader(temp_file_path)
                    else:
                        st.warning(f"Unsupported file type: {file_extension}")
                        os.remove(temp_file_path)
                        continue
                    
                    documents = loader.load()
                    
                    # Add source metadata
                    for doc in documents:
                        doc.metadata["source"] = file.name
                    
                    st.session_state["uploaded_texts"].extend(documents)
                    new_files = True
                    logger.info(f"Processed file: {file.name}")
                    
                    # Clean up temp file
                    os.remove(temp_file_path)
                    
                except Exception as e:
                    logger.error(f"Error processing file {file.name}: {e}")
                    st.error(f"Error processing {file.name}: {e}")
        
        if new_files:
            # Reset vector store when new files are added
            st.session_state["vector_store"] = None
            st.session_state["chain"] = None


def setup_vector_store():
    """
    Create or retrieve vector store with embeddings.
    Uses RecursiveCharacterTextSplitter for better semantic chunking.
    """
    if not st.session_state["uploaded_texts"]:
        return None
    
    if st.session_state["vector_store"] is not None:
        return st.session_state["vector_store"]
    
    try:
        with st.spinner("Creating vector store..."):
            # Use RecursiveCharacterTextSplitter for better chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            text_chunks = text_splitter.split_documents(st.session_state["uploaded_texts"])
            logger.info(f"Created {len(text_chunks)} text chunks")
            
            # Initialize embeddings
            embedding = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"}
            )
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=text_chunks,
                embedding=embedding,
                persist_directory=settings.persist_directory
            )
            
            st.session_state["vector_store"] = vector_store
            logger.info("Vector store created successfully")
            return vector_store
            
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        st.error(f"Failed to create vector store: {e}")
        return None


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vector_store):
    """
    Create RAG chain using LCEL (LangChain Expression Language).
    Modern pattern with pipe syntax for better composability.
    """
    try:
        llm = get_llm()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": settings.retrieval_k}
        )
        
        # Define the prompt template
        template = """You are a helpful AI assistant. Use the following context to answer the user's question.
If you don't know the answer based on the context, say so. Don't make up information.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create LCEL chain with pipe syntax
        rag_chain = (
            RunnableParallel({
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: format_chat_history(st.session_state["messages"])
            })
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.info("RAG chain created successfully using LCEL")
        return rag_chain
        
    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        st.error(f"Failed to create RAG chain: {e}")
        return None


def format_chat_history(messages: List) -> str:
    """Format chat history for the prompt."""
    if not messages:
        return "No previous conversation."
    
    formatted = []
    for msg in messages[-6:]:  # Last 3 exchanges
        role = "Human" if msg["is_user"] else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def display_chat():
    """Display chat interface and handle user interactions."""
    if st.session_state["chain"] is None:
        st.info("👆 Please upload documents in the sidebar to start chatting!")
        return
    
    # Chat input
    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question about your documents:",
                placeholder="What is this document about?",
                key="input"
            )
            col1, col2 = st.columns([1, 5])
            with col1:
                submit_button = st.form_submit_button(label="Send 📤")
        
        if submit_button and user_input:
            try:
                # Add user message to history
                st.session_state["messages"].append({
                    "content": user_input,
                    "is_user": True
                })
                
                with st.spinner("Thinking..."):
                    # Invoke the chain
                    response = st.session_state["chain"].invoke(user_input)
                    
                    # Add assistant response to history
                    st.session_state["messages"].append({
                        "content": response,
                        "is_user": False
                    })
                    
                    logger.info(f"Question: {user_input[:50]}... | Answer: {response[:50]}...")
                    
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                st.error(f"Error: {e}")
    
    # Display chat history
    if st.session_state["messages"]:
        st.markdown("---")
        chat_container = st.container()
        with chat_container:
            for idx, msg in enumerate(st.session_state["messages"]):
                message(
                    msg["content"],
                    is_user=msg["is_user"],
                    key=f"msg_{idx}",
                    avatar_style="thumbs" if msg["is_user"] else "bottts"
                )


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    local_css()
    
    # Header
    st.title(f"🤖 {settings.app_title}")
    st.markdown(
        '<div class="info-text">📚 Upload your documents and ask questions. '
        'Powered by LangChain with LCEL patterns.</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.title("📁 Document Management")
        st.markdown("---")
        
        # Configuration display
        with st.expander("⚙️ Configuration", expanded=False):
            st.write(f"**Model:** {settings.default_model}")
            st.write(f"**Temperature:** {settings.model_temperature}")
            st.write(f"**Chunk Size:** {settings.chunk_size}")
            st.write(f"**Retrieval K:** {settings.retrieval_k}")
        
        st.markdown("---")
        
        # File upload
        initialize_session_state()
        upload_and_process_files()
        
        # Display uploaded files
        if st.session_state["uploaded_texts"]:
            st.success(f"✅ {len(set(doc.metadata.get('source', '') for doc in st.session_state['uploaded_texts']))} file(s) loaded")
            
            if st.button("🗑️ Clear All Documents"):
                st.session_state["uploaded_texts"] = []
                st.session_state["vector_store"] = None
                st.session_state["chain"] = None
                st.session_state["messages"] = []
                st.rerun()
    
    # Setup vector store and chain
    if st.session_state["uploaded_texts"] and st.session_state["chain"] is None:
        vector_store = setup_vector_store()
        if vector_store:
            st.session_state["chain"] = create_rag_chain(vector_store)
    
    # Display chat interface
    display_chat()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888;">Built with LangChain & Streamlit | '
        'Using LCEL patterns for modern AI applications</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An unexpected error occurred: {e}")
