import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM   # ⬅️ use Ollama instead of ChatOpenAI

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save to temp
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing document...")

    # 1. Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # 3. Embeddings (local, via sentence-transformers)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Vector store (Chroma, in-memory for now)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5. Local LLM via Ollama (llama3:8b)
    llm = OllamaLLM(model="llama3:8b")   # ⬅️ make sure `ollama pull llama3:8b` is done

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # 6. Chat UI
    user_question = st.text_input("Ask a question about your PDF")

    if user_question:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": user_question})

        st.subheader("Answer")
        st.write(result["result"])

        st.subheader("Sources")
        for i, doc in enumerate(result["source_documents"]):
            with st.expander(f"Source {i+1} - page {doc.metadata.get('page', 'unknown')}"):
                st.write(doc.page_content[:1000])