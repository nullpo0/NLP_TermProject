# NLP_TermProject
[CNU AI] NLP TermProject

pre-trained LLM에 fine-tuning하여 충남대학교 특화 LLM을 개발하여 챗봇으로 활용하는 프로젝트.

---

### 사용된 모델
1. classifier base model : ```klue/bert-base``` 

    link : https://huggingface.co/klue/bert-base

2. LLM base model : ```MLP-KTLim/llama-3-Korean-Bllossom-8B``` 

    link : https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B

3. embedding model : ```jhgan/ko-sbert-sts```

    link : https://huggingface.co/jhgan/ko-sbert-sts
---

### 사용법
1. python, node.js 설치 필요

2. root directory에서 터미널(bash)을 열고 ```source init.sh```를 통해 가상환경 구축을 해준다.(5~10분 소요, 약 6~7GB)

3. 완료되면 터미널을 닫고 다시 시작해준다.

4. 그 다음 ```source chatbot.sh```를 통해 실행시켜준다. (초기 실행 시 시간이 조금 걸림)

실행순서는

1. data 폴더의 test_cls.json, test_realtime.json을 입력으로 하여 각각 fine-tuning된 LLM에 RAG 미적용, RAG 적용된 출력을 output에 저장 (총 2분가량)

2. 서버가 열리고 ui가 띄워짐. 채팅을 보내면 그에 맞게 챗봇이 답변을 함.(답변 당 10~20초)
---

### 주의사항

1. 해당 프로젝트는 GPU사용을 기준으로 구현되었음.

2. 요구 디스크 용량은 총 6.8 ~ 7GB (모델까지 합하면 약 24 ~ 25GB)

3. 요구 메모리 용량은 약 1GB ~ 2GB