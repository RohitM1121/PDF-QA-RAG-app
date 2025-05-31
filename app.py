import streamlit as st
import os
import tempfile

# Ensure all your RAG imports are here
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Global variables or configurable parameters
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_LENGTH_LLM = 512

# --- RAG Functions
@st.cache_resource
def load_embedding_model():
    st.info("Loading embedding model (this may take a moment)...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_llm_pipeline():
    st.info(f"Loading LLM pipeline: {LLM_MODEL_NAME} (this may take a while)...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=MAX_LENGTH_LLM)
    return HuggingFacePipeline(pipeline=pipe)

def process_pdf_and_create_vectorstore(pdf_path: str):
    """
    Loads a PDF, splits text, and creates a FAISS vector store.
    """
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(documents)

    embedding_model = load_embedding_model()
    vectordb = FAISS.from_documents(docs, embedding_model)
    return vectordb

def get_rag_answer(query: str, vectordb) -> dict:
    """
    Given a query and a FAISS vector store, retrieves relevant information
    and generates an answer using the LLM.
    """
    llm = load_llm_pipeline()
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    response = qa_chain.invoke({"query": query})
    return response


# Streamlit UI
st.set_page_config(page_title="PDF Q&A with RAG", layout="centered")

st.title("PDF Q&A with Retrieval Augmented Generation (RAG)")
st.markdown("""
Upload a PDF document and ask questions about its content.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file is not None:
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success(f"PDF uploaded successfully: {uploaded_file.name}")

    # Process PDF and create vector store
    with st.spinner("Processing PDF and creating knowledge base... This might take a few moments."):
        if 'vectordb' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
            st.session_state.vectordb = process_pdf_and_create_vectorstore(pdf_path)
            st.session_state.uploaded_file_name = uploaded_file.name # Store the name to check for new uploads
            st.success("Knowledge base created!")
        else:
            st.info("Using existing knowledge base for the uploaded PDF.")

    # Remove the temporary file after processing
    os.unlink(pdf_path)

    # Question Input
    if 'vectordb' in st.session_state:
        query = st.text_input("Ask a question about the document:")

        if query:
            with st.spinner("Getting answer..."):
                try:
                    response = get_rag_answer(query, st.session_state.vectordb)
                    st.subheader("Answer:")
                    st.write(response['result'])

                    if response.get('source_documents'):
                        st.subheader("Source Documents:")
                        for i, doc in enumerate(response['source_documents']):
                            st.write(f"**Document {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                            st.info(doc.page_content[:500] + "...") # Show first 500 chars
                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    st.info("Please try re-uploading the PDF or rephrasing your question.")
    else:
        st.warning("Please upload a PDF document to start asking questions.")
else:
    st.info("Awaiting PDF upload...")

st.markdown("""
---
*Note: The first time you run this application, it will download the embedding and LLM models, which may take a while. Subsequent runs will be faster.*
""")
