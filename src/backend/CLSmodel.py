import torch
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel

class CLSModel:
    def __init__(self, adapter_path="./model/classifier_model"):
        self.tokenizer = BertTokenizer.from_pretrained(adapter_path)
        self.model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=5)
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.label = ["졸업요건", "학교 공지사항", "학사일정", "식단 안내", "통학/셔틀버스"]
        
    def classification(self, question, string=True):
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predict = logits.argmax(dim=-1).item()
        
        label = self.label[predict]
        
        if string is True:
            return label
        else:
            return predict