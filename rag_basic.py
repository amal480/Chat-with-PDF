from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

llm = OllamaLLM(model="llama3:8b")
loader = PyPDFLoader("temp.pdf")
docs = loader.load()  # list of Document objects

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)
# print(len(chunks), "chunks")
# print(chunks[10].page_content[:500])


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="pdf_rag"
)

question = "What is attention?"
retrieved_docs = vectorstore.similarity_search(question, k=4)

for i, d in enumerate(retrieved_docs):
    print(f"\n--- Retrieved chunk {i} ---")
    print(d.page_content[:400])


def answer_question(question: str):
    # 2) Retrieve relevant chunks from Chroma
    retrieved = vectorstore.similarity_search(question, k=4)
    context_text = "\n\n".join(d.page_content for d in retrieved)

    # 3) Build a prompt that tells the model to ONLY use the context
    prompt = f"""
You are a helpful assistant answering questions based ONLY on the context below.

Context:
{context_text}

Question:
{question}

If the answer is not clearly in the context, say:
"I don't know from the document."
"""

    # 4) Call the local LLM via Ollama
    response = llm.invoke(prompt)

    return response

# Test it
print(answer_question("What is the main idea of this document?"))

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # stuff = put all retrieved docs into prompt
    return_source_documents=True
)

result = qa_chain.invoke({"query": "What is the objective of this paper?"})
print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(doc.metadata, "=>", doc.page_content[:200])
