import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import json
from tqdm import tqdm

def gen_answer(instruction):
    PROMPT = '''너는 충남대학교 인공지능 챗봇 상담사야. 질문에 대해 짧게 한 두문장으로 답변해.'''
    instruction = instruction

    messages = [
        {"role": "system", "content": f"{PROMPT}"},
        {"role": "user", "content": f"{instruction}"}
        ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    
    return outputs[0][input_ids.shape[-1]:]

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

tokenizer = AutoTokenizer.from_pretrained("./model/LLMFT_model")

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda:0",
    torch_dtype=torch.float16,
    load_in_4bit=True
    )

model = PeftModel.from_pretrained(base_model, "./model/LLMFT_model")
model.eval()

test_chat = pd.read_json("./data/test_chat.json")

dic = []

for question in tqdm(test_chat['user']):
    answer = tokenizer.decode(gen_answer(question), skip_special_tokens=True)
    dic.append({"user": question, "model": answer})
    
with open("./outputs/chat_outputs.json", "w", encoding="utf-8") as f:
    json.dump(dic, f, ensure_ascii=False, indent=4)