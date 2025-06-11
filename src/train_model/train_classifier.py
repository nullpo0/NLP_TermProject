import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

class QuestionDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer(questions, truncation=True, padding=True, max_length=max_len)
        self.labels = labels
        
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tokenizer['input_ids'][idx]),
            'attention_mask': torch.tensor(self.tokenizer['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx]),
        }

    def __len__(self):
        return len(self.labels)   


data = pd.read_json("data/q.json")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["question"].tolist(),
    data["label"].tolist(),
    test_size=0.2,
    random_state=42
)

model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

train_dataset = QuestionDataset(train_texts, train_labels, tokenizer)
val_dataset = QuestionDataset(val_texts, val_labels, tokenizer)  
  
training_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    weight_decay=0.01,
    learning_rate=0.001,
    save_strategy="no"
)

# 1

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("model/classifier_model")
