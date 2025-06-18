import pandas as pd
import json
from tqdm import tqdm
from FTmodel import FTModel
from retriever import retriever

model = FTModel()

test_chat = pd.read_json("./data/test_chat.json")
test_realtime = pd.read_json("./data/test_realtime.json")

dic1 = []
dic2 = []

for question in tqdm(test_chat['user']):
    prompt1 = "당신은 충남대학교 인공지능 챗봇입니다. 사용자의 질문에 대해 짧게 한 두 문장으로 답변하세요."
    
    answer = model.tokenizer.decode(model.generate_answer(prompt=prompt1, question=question), skip_special_tokens=True)
    dic1.append({"user": question, "model": answer})
    
with open("./outputs/chat_outputs.json", "w", encoding="utf-8") as f:
    json.dump(dic1, f, ensure_ascii=False, indent=4)
    
    
for question in tqdm(test_realtime['user']):
    retrieve_result = retriever(question)
    
    prompt2 = f"""
            당신은 충남대학교 인공지능 챗봇입니다. 사용자의 질문에 대해 짧게 한 두 문장으로 답변하세요. 아래 자료를 참고하여 사용자의 질문에 대해 짧게 한 두 문장으로 답변하세요.
            
            [자료]
            {retrieve_result}
        """
    
    answer = model.tokenizer.decode(model.generate_answer(prompt=prompt2, question=question), skip_special_tokens=True)
    dic2.append({"user": question, "model": answer})

with open("./outputs/realtime_outputs.json", "w", encoding="utf-8") as f:
    json.dump(dic2, f, ensure_ascii=False, indent=4)
    