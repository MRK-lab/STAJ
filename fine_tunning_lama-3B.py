import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from unsloth import FastLanguageModel


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-bnb-4bit",  #alternatif: Llama-3.2-1B-Instruct
    max_seq_length = 128,
    dtype = torch.float16,# desteklenmiyorsa: torch.bfloat16
    load_in_4bit = True
)

df = pd.read_csv("dataset_06.csv")
print(df.head())

dataset = Dataset.from_pandas(df)#=>{'input': 'Merhaba', 'output': 'Hello'}

def preprocess_function(examples):
    inputs = examples["input"]
    outputs = examples["output"]

    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )

    labels = tokenizer(
        outputs,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=True,
    hub_model_id="BeyzaNurYldrmm/06_fine",
    hub_token="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"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

model.push_to_hub("BeyzaNurYldrmm/06_fine", use_auth_token=True)
tokenizer.push_to_hub("BeyzaNurYldrmm/06_fine", use_auth_token=True)
