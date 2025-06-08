from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Data(BaseModel):
    message: str
    
@app.get("/")
async def root():
    return {"message": "hello"}
    
@app.post("/chat")
async def chat(data: Data):
    length = len(data.message)
    return {"len": f'length: {length}'}

