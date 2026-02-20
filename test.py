import torch
import streamlit as st
import tempfile
import os

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import LlamaTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline


from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# Khoi tao Session State de dam bao StreamLit hoat dong on dinh va ko bi tai lai model
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None


# Ham tai embedding Model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")


# Ham tair Large Language Model
@st.cache_resource
def load_llm():
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True,
        device_map=None,
        # Sua de chay voi pytorch <2.6
        use_safetensors=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )
    return HuggingFacePipeline(pipeline=model_pipeline)

# Ham xu ly PDF
def process_pdf(upload_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(upload_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size= 1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount= 95,
        min_chunk_size= 500,
        add_start_index= True
    )

    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(documents = docs, embedding =st.session_state.embeddings)
    retriever = vector_db.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | st.session_state.llm
            | StrOutputParser()
    )

    os.unlink(tmp_file_path)
    return rag_chain, len(docs)


# Cau hinh tran tieu de
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF RAG Assistant")
st.markdown("""
5 **Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF b

ằng tiếng Việt**

6 **Cách sử dụng đơn giản:**
7 1. **Upload PDF** -> Chọn file PDF từ máy tính và nhấn "Xử lý PDF"
8 2. **Đặt câu hỏi** -> Nhập câu hỏi về nội dung tài liệu và nhận câu

trả lời ngay lập tức

9 ---
10 """)

# Tai models

if not st.session_state.model_loaded:
    st.info("Dang tai model....")
    st.session_state.embeddings = load_embeddings()
    st.session_state.llm = load_llm()
    st.session_state.model_loaded = True
    st.success("Model da san sang!")
    st.rerun()


uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file and st.button("Xu ly PDF"):
    with st.spinner("Dang xu ly...."):
        st.session_state.rag_chain, num_chunks= process_pdf(uploaded_file)
        st.success("complete")


# giao dien hoi dao

if st.session_state.rag_chain:
    question = st.text_input("Dat cau hoi: ")
    if question:
        with st.spinner("Dang xu ly...."):
            output = st.session_state.rag_chain.invoke(question)
            answer = output.split("Answer:")[1].strip() if "Answer" in output else output.strip()
            st.write("tra loi")
            st.write(answer)