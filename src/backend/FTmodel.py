import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class FTModel:
    def __init__(self, adapter_path="./model/LLMFT_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            "MLP-KTLim/llama-3-Korean-Bllossom-8B",
            device_map="cuda:0",
            torch_dtype=torch.float16,
            load_in_4bit=True
            )
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
    def generate_answer(self, prompt, question):
        PROMPT = prompt
        instruction = question

        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{instruction}"}
            ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=128,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        return outputs[0][input_ids.shape[-1]:]
    