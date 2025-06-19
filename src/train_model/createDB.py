from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

category_urls = {
    "학교 공지사항": [
        "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0701&site_dvs_cd=kr&menu_dvs_cd=0701",
        "https://plus.cnu.ac.kr/_prog/_board/?code=sub07_0702&site_dvs_cd=kr&menu_dvs_cd=0702",
        "https://homepage.cnu.ac.kr/ai//board/notice.do"
    ],
    "학사일정": [
        "https://plus.cnu.ac.kr/_prog/academic_calendar/?site_dvs_cd=kr&menu_dvs_cd=05020101&year=2025"
    ],
    "식단 안내": [
        "https://mobileadmin.cnu.ac.kr/food/index.jsp?searchYmd=2025.06.20&searchLang=OCL04.10&searchView=cafeteria&searchCafeteria=OCL03.02&Language_gb=OCL04.10",
        "https://mobileadmin.cnu.ac.kr/food/index.jsp?searchYmd=2025.06.19&searchLang=OCL04.10&searchView=cafeteria&searchCafeteria=OCL03.02&Language_gb=OCL04.10",
        "https://mobileadmin.cnu.ac.kr/food/index.jsp?searchYmd=2025.06.18&searchLang=OCL04.10&searchView=cafeteria&searchCafeteria=OCL03.02&Language_gb=OCL04.10"
    ],
    "통학/셔틀버스": [
        "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html"
    ]
}

all_docs = []

for category, urls in category_urls.items():
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
            docs = splitter.split_documents(documents)

            for doc in docs:
                doc.metadata["category"] = category
                doc.metadata["source_url"] = url

            all_docs.extend(docs)

        except Exception as e:
            print(f"{url} 로드 실패: {e}")
            
vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding_model,
    persist_directory="./src/backend/vectorDB"
)
vectorstore.persist()

