# rag/pdf_processor.py

import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker


class PDFProcessor:

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def process(self, uploaded_file):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = SemanticChunker(
            embeddings=self.embeddings,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=500,
            add_start_index=True
        )

        docs = splitter.split_documents(documents)

        os.unlink(tmp_path)

        return docs