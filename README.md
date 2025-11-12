# 🧠 Local RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot built using **LangChain**, **Hugging Face**, and **Streamlit** — designed to let you **upload PDFs** and **ask questions** directly from the document content.

---

## 🚀 Features

- 📄 Upload up to 5 PDF files  
- 🧩 Hybrid retrieval: combines **semantic search (FAISS)** and **keyword-based search**  
- 🧠 Local LLM inference using **google/flan-t5-base**  
- 💬 Conversational memory for context-aware chats  
- 🎨 Modern dark UI with smooth animations  
- ⚙️ Adjustable chunk size, overlap, and retrieval depth  
- 🧾 Built-in relevance filtering and fallback snippet extraction  

---

## 🖼️ Demo Screenshot

*(Optional — add your image in `assets/` folder and update path below)*  
![Local RAG Chatbot Demo](assets/chatbot_demo.png)

---

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| **Streamlit** | Web app UI |
| **LangChain** | Retrieval and memory management |
| **FAISS** | Vector store for semantic search |
| **HuggingFace Transformers** | Model pipeline (`flan-t5-base`) |
| **PyPDFLoader** | PDF parsing |
| **Sentence Transformers** | Embedding generation (`all-MiniLM-L6-v2`) |


   git clone https://github.com/<your-username>/local-rag-chatbot.git
   cd local-rag-chatbot
