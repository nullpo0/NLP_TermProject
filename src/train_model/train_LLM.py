import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers
from datasets import load_dataset
from peft import prepare_model_for_kbit_training


model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer_id = model_id

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    resume_download=True,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

for param in model.parameters():
  param.requires_grad = False 
  if param.ndim == 1:
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()

config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

dataset = load_dataset(path='./data', data_files='qa.json')

def preprocess_chat(example):
    messages = [
        {"role": "system", "content": "당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 친절하게 답변해주세요."},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )[0]

    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}

tokenized_dataset = dataset.map(preprocess_chat, remove_columns=["question", "answer"])
print(tokenized_dataset)
print(tokenized_dataset['train'][0])

train_args = transformers.TrainingArguments(
    output_dir='./model/LLMFT_model',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
)

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False

trainer.train()
model.save_pretrained("./model/LLMFT_model")
tokenizer.save_pretrained("./model/LLMFT_model")
