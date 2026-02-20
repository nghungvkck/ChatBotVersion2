# app.py

import streamlit as st

from core.llm_loader import LLMFactory
from core.embedding_loader import EmbeddingLoader
from rag.pdf_processor import PDFProcessor
from rag.rag_chain import RAGPipeline


# Session State
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


@st.cache_resource
def load_models():
    embeddings = EmbeddingLoader().load()
    llm = LLMFactory().load()
    return embeddings, llm


# UI Config
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📄 PDF RAG Assistant")

# Load model
if not st.session_state.model_loaded:
    st.info("Đang tải model...")
    embeddings, llm = load_models()

    st.session_state.embeddings = embeddings
    st.session_state.llm = llm
    st.session_state.model_loaded = True
    st.success("Model sẵn sàng!")
    st.rerun()


# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file and st.button("Xử lý PDF"):
    with st.spinner("Đang xử lý..."):
        processor = PDFProcessor(st.session_state.embeddings)
        docs = processor.process(uploaded_file)

        rag = RAGPipeline(
            llm=st.session_state.llm,
            embeddings=st.session_state.embeddings
        )

        rag.build(docs)
        st.session_state.rag_pipeline = rag

        st.success(f"Hoàn thành! Số chunk: {len(docs)}")


# Chat
if st.session_state.rag_pipeline:
    question = st.text_input("Đặt câu hỏi:")
    if question:
        with st.spinner("Đang trả lời..."):
            answer = st.session_state.rag_pipeline.ask(question)
            st.write("### Trả lời:")
            st.write(answer)