# rag/rag_chain.py

from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RAGPipeline:

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.chain = None

    def build(self, documents):

        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        retriever = vector_db.as_retriever()

        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": retriever | format_docs,
             "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return self.chain

    def ask(self, question: str):
        return self.chain.invoke(question)