# rag/pdf_processor.py

import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

# ===============================
# class tiến trình xử lý file PDF
# ===============================
class PDFProcessor:

    def __init__(self, embeddings):
        self.embeddings = embeddings

# Load file pdf
    def process(self, uploaded_file):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

# Khởi tạo bộ tách văn bản
# Giúp chia văn bản thành các phần có ý nghĩa và bảo đảm sự liên lết ngữ nghĩa trong mỗi chuck
        splitter = SemanticChunker(
            embeddings=self.embeddings,
            buffer_size=1,  # Số câu gom trước khi tách
            breakpoint_threshold_type="percentile",  # Dựa trên độ tương đồng (percentile)
            breakpoint_threshold_amount=95,  # Ngưỡng % để cắt đoạn
            min_chunk_size=500,  # Kích thước tối thiểu mỗi chunk
            add_start_index=True  # Đánh dấu vị trí chunk trong văn bản
        )

        docs = splitter.split_documents(documents)

        os.unlink(tmp_path)

        return docs