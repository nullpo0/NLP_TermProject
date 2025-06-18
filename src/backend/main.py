from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
from pydantic import BaseModel
from retriever import retriever
from FTmodel import FTModel


class Data(BaseModel):
    message: str
    
model_ready = False

model = None

def load_model():
    global model, model_ready
    model = FTModel(adapter_path="../../model/LLMFT_model")
    model_ready = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=load_model).start()
    yield
    

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
async def get_status():
    return {"ready": model_ready}

@app.post("/chat")
async def chat(data: Data):
    # retrieve_result = retriever(data.message)
    
    # PROMPT = f"""
    #         당신은 충남대학교 인공지능 챗봇입니다. 사용자의 질문에 대해 짧게 한 두 문장으로 답변하세요. 아래 자료를 참고하여 사용자의 질문에 대해 짧게 한 두 문장으로 답변하세요.
            
    #         [자료]
    #         {retrieve_result}
    #     """
    
    PROMPT = "당신은 충남대학교 인공지능 챗봇입니다. 사용자의 질문에 대해 짧게 한 두 문장으로 답변하세요."
    answer = model.tokenizer.decode(model.generate_answer(prompt=PROMPT, question=data.message), skip_special_tokens=True)
    return {"answer": answer}