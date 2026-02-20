# core/embedding_loader.py

from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL

class EmbeddingLoader:

    def __init__(self):
        self.model_name = EMBEDDING_MODEL

    def load(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_name
        )
