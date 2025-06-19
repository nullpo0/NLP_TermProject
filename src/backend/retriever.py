from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class Retriever:
    def __init__(self, persist_directory="./src/backend/vectorDB"):
        self.embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )

    def retrieve(self, question, label):
        print(label)
        if label == "졸업요건":
            return "없음"
        retrieve_result = self.db.similarity_search(question, k=1, filter={"category": label})
        return retrieve_result[0].page_content
