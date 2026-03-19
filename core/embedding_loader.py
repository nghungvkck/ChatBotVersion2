# core/embedding_loader.py
'''
Khởi tạo vector database
Sau khi tách văn bản thành các chuck có ý nghĩa và biểu diễn 
chúng dưới dạng các vector với mô hình embedding, bước tiếp theo 
là khởi tạo một vector database để lưu trữ các vector của các chuck này.
'''
from langchain_huggingface import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL

class EmbeddingLoader:

    def __init__(self):
        self.model_name = EMBEDDING_MODEL

    def load(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_name
        )
