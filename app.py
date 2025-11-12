import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from datetime import datetime
from typing import List
import pytz
import time
import re
import numpy as np

st.set_page_config(page_title="Local RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    /* Chat container */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 25px;
        border-radius: 20px;
        background: rgba(30, 30, 30, 0.95);
        border: 2px solid #3498db;
        box-shadow: 0 8px 32px rgba(52, 152, 219, 0.3);
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Chat bubbles */
    .chat-row {
        display: flex;
        margin-bottom: 18px;
        animation: slideIn 0.4s ease-out;
    }
    
    .chat-bubble {
        padding: 15px 22px;
        border-radius: 18px;
        max-width: 75%;
        word-wrap: break-word;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        margin-right: 10px;
        border-bottom-right-radius: 4px;
    }
    
    .bot-bubble {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: #ffffff;
        margin-right: auto;
        margin-left: 10px;
        border-bottom-left-radius: 4px;
    }
    
    /* Timestamps */
    .timestamp {
        font-size: 11px;
        color: #95a5a6;
        margin: 2px 12px;
        font-style: italic;
    }
    
    .user-timestamp {
        text-align: right;
    }
    
    .bot-timestamp {
        text-align: left;
    }
    
    /* Header styling */
    .header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .header img {
        width: 50px;
        height: 50px;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
    }
    
    .header h1 {
        color: #ffffff;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Query section */
    .query-section {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .query-section p {
        color: #ecf0f1;
        font-size: 17px;
        margin: 0;
        font-weight: 500;
    }
    
    /* Status boxes */
    .status-box {
        padding: 12px;
        border-radius: 10px;
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        color: white;
        font-size: 14px;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(39, 174, 96, 0.3);
    }
    
    .error-box {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        box-shadow: 0 4px 10px rgba(231, 76, 60, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        box-shadow: 0 4px 10px rgba(243, 156, 18, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.6;
        }
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: rgba(30, 30, 30, 0.5);
        border-radius: 10px;
    }
    
    /* Loading animation */
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>Local RAG Chatbot</h1>
</div>
<div class="query-section">
    <p>📄 Upload PDFs and ask intelligent questions</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Documents")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload PDF files (Max 5)", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload up to 5 PDF files for analysis"
    )
    
    # Validate max 5 PDFs
    if uploaded_file and len(uploaded_file) > 5:
        st.error("⚠️ Maximum 5 PDFs allowed. Please remove some files.")
        uploaded_file = uploaded_file[:5]
    
    if uploaded_file:
        st.success(f"✅ {len(uploaded_file)} PDF(s) uploaded")
        for idx, file in enumerate(uploaded_file, 1):
            st.info(f"{idx}. {file.name}")
    
    st.markdown("---")
    
    # Retrieval settings
    st.header("⚙️ Retrieval Settings")
    search_k = st.slider("Number of chunks to retrieve", 3, 10, 6, help="More chunks = better context but slower")
    chunk_size = st.slider("Chunk size", 300, 800, 400, step=50, help="Smaller chunks = more precise retrieval")
    chunk_overlap = st.slider("Chunk overlap", 50, 200, 100, step=25, help="Higher overlap = better context continuity")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
        st.session_state.chat_history = []
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.success("✨ Chat history cleared!")
        st.rerun()
    

def clean_lab_text(text: str) -> str:
    pattern = re.compile(r"(ng/mL|mg/dL)\s+(\d+\.\d+)\s*-\s*(\d+\.\d+)(\d+\.\d+)")
    def replacer(m):
        unit = m.group(1)
        ref_low = m.group(2)
        ref_high = m.group(3)
        result = m.group(4)
        return f"**Result**: {result} {unit}, **Reference Range**: {ref_low} – {ref_high} {unit}"
    return pattern.sub(replacer, text)

def is_relevant_response(response: str, question: str, source_docs) -> tuple:
    """
    Enhanced relevance checking that validates against source documents
    Returns: (is_relevant: bool, confidence_score: float)
    """
    if not source_docs:
        return False, 0.0
    
    # Check for generic/evasive responses
    irrelevant_phrases = [
        "i don't know",
        "no information",
        "cannot answer",
        "not mentioned",
        "no context",
        "insufficient information",
        "not provided",
        "unable to answer",
        "i apologize",
        "i'm sorry"
    ]
    
    response_lower = response.lower()
    question_lower = question.lower()
    
    # Check for very short or irrelevant responses
    if len(response.split()) < 3:
        return False, 0.1
    
    # Check for evasive phrases
    for phrase in irrelevant_phrases:
        if phrase in response_lower:
            return False, 0.2
    
    # Check if response contains keywords from question
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    response_words = set(re.findall(r'\b\w+\b', response_lower))
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'when', 'where', 'who', 'how', 'why'}
    question_words -= stop_words
    
    # Calculate keyword overlap
    overlap = len(question_words & response_words) / max(len(question_words), 1)
    
    # Check if source documents contain relevant content
    source_text = " ".join([doc.page_content.lower() for doc in source_docs])
    source_relevance = sum(1 for word in question_words if word in source_text) / max(len(question_words), 1)
    
    confidence = (overlap + source_relevance) / 2
    
    return confidence > 0.2, confidence

def extract_keywords(text: str) -> list:
    """Extract important keywords from query"""
    # Remove common words and extract meaningful terms
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'when', 'where', 
                  'who', 'how', 'why', 'which', 'about', 'tell', 'me', 'please', 'can', 'you'}
    
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords

class HybridRetriever(BaseRetriever):
    """Combines semantic search with keyword-based search for better retrieval"""
    
    vector_store: object  # Changed from FAISS to object for compatibility
    documents: List[Document]
    k: int = 6
    
    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid'  # Prevent extra fields
    
    def keyword_search(self, query: str, k: int = None) -> List[Document]:
        """Perform keyword-based search on documents"""
        if k is None:
            k = self.k
        
        keywords = extract_keywords(query)
        scored_docs = []
        
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            score = 0
            
            # Score based on keyword presence and frequency
            for keyword in keywords:
                count = content_lower.count(keyword)
                if count > 0:
                    # Boost score for multiple occurrences
                    score += count * (len(keyword) / max(len(content_lower), 1))
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:k]]
    
    def semantic_search(self, query: str, k: int = None) -> List[Document]:
        """Perform semantic vector search"""
        if k is None:
            k = self.k
        
        docs = self.vector_store.similarity_search(query, k=k)
        return docs
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Combine keyword and semantic search with deduplication"""
        k = self.k
        
        # Get results from both methods
        keyword_docs = self.keyword_search(query, k=k)
        semantic_docs = self.semantic_search(query, k=k)
        
        # Combine and deduplicate based on content
        seen_content = set()
        combined_docs = []
        
        # Prioritize keyword matches (more precise)
        for doc in keyword_docs:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as identifier
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined_docs.append(doc)
        
        # Add semantic matches
        for doc in semantic_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined_docs.append(doc)
        
        return combined_docs[:k]


documents = []
chunks = []
pdf_loaded = False

if uploaded_file and len(uploaded_file) <= 5:
    with st.spinner("🔄 Processing PDF(s)..."):
        for file in uploaded_file:
            try:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                
                # Clean and process documents
                for doc in docs:
                    doc.page_content = clean_lab_text(doc.page_content)
                    # Add metadata for better tracking
                    doc.metadata['source_file'] = file.name
                
                documents.extend(docs)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        if documents:
            pdf_loaded = True
            st.success(f"✅ Successfully processed {len(documents)} pages!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )

if documents:
    # Use RecursiveCharacterTextSplitter with optimized settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # More granular splitting
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    
    # Store full documents for keyword search
    st.session_state.full_documents = chunks
    
    @st.cache_resource
    def get_embeddings():
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    embeddings = get_embeddings()
    
    # Build vector store with improved indexing
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Initialize hybrid retriever with keyword arguments only (Pydantic requirement)
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        documents=chunks,
        k=search_k
    )
    
    @st.cache_resource
    def get_llm():
        hf_pipeline = pipeline(
            "text2text-generation", 
            model="google/flan-t5-base", 
            max_length=512,
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.3,
            do_sample=True
        )
        return HuggingFacePipeline(pipeline=hf_pipeline)
    
    llm = get_llm()
    
    custom_template = """
    You are a knowledgeable assistant that must answer strictly based on the given context.

    Context:
    {context}

    Instructions:
    1. Use only the information present in the above context to answer the question.
    2. If the question cannot be answered completely or partially from the context, respond only with:
   "I couldn't find that information in the provided documents."
    3. Do not generate, guess, or include any external knowledge.
    4. Keep your answers factual, concise, and directly related to the question.
    5. If the question is unrelated to the context (e.g., personal, general, or out-of-scope), reply with:
   "This question is outside the scope of the provided documents."

    Question:
    {question}

    Answer:
    """

    
    CUSTOM_PROMPT = PromptTemplate(
        template=custom_template, input_variables=["context", "question"]
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,  # Now properly inherits from BaseRetriever
        memory=st.session_state.memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
    )

# Simple responses
simple_responses = {
    "thank you": "You're welcome! 😊 Feel free to ask more questions!",
    "thanks": "My pleasure! 😄 How else can I help?",
    "hello": "Hi there! 👋 Upload a PDF and ask me questions about it!",
    "hi": "Hello! 😄 What would you like to know?",
    "hey": "Hey! 👋 Ready to help you explore your documents!",
    "bye": "See you later! Explore more! 👋",
    "goodbye": "Goodbye! Feel free to return anytime! 😊"
}

# Chat Input
user_input = st.chat_input("💬 Type your message here...")

if user_input:
    ist = pytz.timezone("Asia/Kolkata")
    current_time = datetime.now(ist).strftime("%H:%M:%S")
    st.session_state.chat_history.append(("user", user_input, current_time))

    with st.spinner("🤔 Thinking..."):
        # Check for simple responses first
        user_input_lower = user_input.lower().strip()
        bot_message = simple_responses.get(user_input_lower, None)
        
        if not bot_message:
            if not documents:
                bot_message = "⚠️ Please upload at least one PDF document to ask questions. I can only answer questions based on the uploaded documents."
            else:
                try:
                    # Get response from QA chain with hybrid retrieval
                    response = qa_chain.invoke({"question": user_input})
                    bot_message = response["answer"]
                    source_docs = response.get("source_documents", [])
                    
                    # Enhanced relevance checking
                    is_relevant, confidence = is_relevant_response(bot_message, user_input, source_docs)
                    
                    if not is_relevant or confidence < 0.3:
                        # Try to extract specific information from source docs
                        if source_docs:
                            keywords = extract_keywords(user_input)
                            relevant_snippets = []
                            
                            for doc in source_docs:
                                content_lower = doc.page_content.lower()
                                for keyword in keywords:
                                    if keyword in content_lower:
                                        # Extract sentence containing keyword
                                        sentences = doc.page_content.split('.')
                                        for sent in sentences:
                                            if keyword in sent.lower():
                                                relevant_snippets.append(sent.strip())
                                                break
                            
                            if relevant_snippets:
                                bot_message = "Based on the documents, here's what I found:\n\n" + "\n\n".join(relevant_snippets[:3])
                            else:
                                bot_message = f"🔍 I couldn't find specific information about '{user_input}' in the uploaded PDF documents. The documents may not contain details about this topic. Try rephrasing your question or ask about different content from the PDFs."
                        else:
                            bot_message = f"🔍 I couldn't find information about '{user_input}' in the uploaded documents. Please ensure your question is related to the PDF content."
                    
                    # Add source reference if relevant
                    if is_relevant and source_docs:
                        source_files = set([doc.metadata.get('source_file', 'Unknown') for doc in source_docs[:2]])
                        if source_files:
                            bot_message += f"\n\n📚 *Source: {', '.join(source_files)}*"
                        
                except Exception as e:
                    bot_message = f"❌ Sorry, I encountered an error: {str(e)}"
        
        current_time = datetime.now(ist).strftime("%H:%M:%S")
        st.session_state.chat_history.append(("bot", bot_message, current_time))
        st.rerun()

with st.container():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align: center; padding: 40px; color: #95a5a6;'>
            <h3>Welcome!!!</h3>
            <p>Upload PDF documents and start asking questions</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for role, msg, timestamp in st.session_state.chat_history:
            if role == "user":
                st.markdown(
                    f"""
                    <div class='chat-row' style='justify-content: flex-end;'>
                        <div class='chat-bubble user-bubble'>{msg}</div>
                    </div>
                    <div class='timestamp user-timestamp'>{timestamp}</div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='chat-row' style='justify-content: flex-start;'>
                        <div class='chat-bubble bot-bubble'>{msg}</div>
                    </div>
                    <div class='timestamp bot-timestamp'>{timestamp}</div>
                    """,
                    unsafe_allow_html=True
                )
    
    st.markdown("</div>", unsafe_allow_html=True)
