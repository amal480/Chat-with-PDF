# 📄 Chat with Your PDF (Offline RAG App with Streamlit, LangChain & Ollama)

A fully local, privacy-friendly PDF Question Answering application built with Streamlit, LangChain, HuggingFace embeddings, ChromaDB, and Ollama (LLaMA 3).  
Upload any PDF and ask questions — the app retrieves relevant content and generates accurate answers using a local LLM.

---

## 🚀 Features

- Upload and process any PDF
- Automatic text chunking with overlap
- Semantic search with vector embeddings
- Local vector database (Chroma)
- Offline LLM inference using Ollama
- Source-aware answers (shows text from PDF)
- Fully local & private (no API keys)

---

## 🛠 Tech Stack

- **UI:** Streamlit  
- **LLM:** Ollama (`llama3:8b`)  
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector Store:** Chroma  
- **Framework:** LangChain  
- **PDF Loader:** PyPDFLoader  

---
