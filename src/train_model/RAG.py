import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# 임베딩 모델 및 QA 모델 설정
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

#  PDF 로드 함수
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    for i, doc in enumerate(docs[:3]):
        print(f"\n PDF Page {i+1}:\n{doc.page_content[:300]}...\n")

    return docs

#  URL 로드 함수 (Document 객체로 반환)
def load_url(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        print(f"\n URL Loaded: {url}\n{text[:300]}...\n")
        return [Document(page_content=text, metadata={"source": url})]
    except Exception as e:
        print(f" Failed to load URL: {url}\nError: {e}")
        return []

#  벡터 저장소 생성 + 문서 조각 확인
def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n Chunk {i+1}:\n{chunk.page_content[:300]}...\n")

    return FAISS.from_documents(chunks, embedding_model)

#  RAG 질의응답 + 관련 문서 확인
def rag_qa(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    print(f"\n관련 문서 ({len(docs)}개):")
    for i, doc in enumerate(docs):
        print(f"\n Related Doc {i+1}:\n{doc.page_content[:300]}...\n")

    context = "\n".join([doc.page_content for doc in docs])
    result = qa_pipeline(question=query, context=context)
    return result["answer"]

# 실행부
if __name__ == "__main__":
    # 여러 PDF 파일 경로
    pdf_paths = [
        "src/train_model/2025.pdf",
        "src/train_model/2025셔틀버스.pdf",
        "src/train_model/2025학사일정.pdf"
    ]

    # 여러 웹페이지 URL
    urls = [
        "https://plus.cnu.ac.kr/_prog/_board/?code=sub05_050204&site_dvs_cd=kr&menu_dvs_cd=050204",
        "https://ai.cnu.ac.kr/ai/board/notice.do",
        "https://mobileadmin.cnu.ac.kr/food/index.jsp"
    ]

    # 문서 로드
    pdf_docs = []
    for path in pdf_paths:
        if os.path.exists(path):
            pdf_docs += load_pdf(path)
        else:
            print(f"파일 없음: {path}")

    url_docs = []
    for link in urls:
        url_docs += load_url(link)

    all_docs = pdf_docs + url_docs

    # 벡터 저장소 생성
    vectorstore = create_vectorstore(all_docs)

    # 질문 예시
    query = "방학 언제야?"
    answer = rag_qa(vectorstore, query)

    print("\n 최종 Answer:")
    print(answer)
