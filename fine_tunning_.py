"""
# json => csv vtden
!pip install psycopg2-binary pandas
import psycopg2
import pandas as pd
import json
from datasets import Dataset

# Bağlantı bilgilerini doldur
conn = psycopg2.connect(
    host="localhost",     # veya sunucu adresi
    database="veritabani_adi",
    user="kullanici_adi",
    password="şifre",
    port=5432             # genelde 5432
)
# SQL sorgusu
sql = "SELECT id, input, output FROM tabloadi"

# pandas ile veriyi çek
df = pd.read_sql_query(sql, conn)
# JSON string'lerini Python dict'e çevir
#df['input'] = df['input'].apply(eval)
#df['output'] = df['output'].apply(eval)

df['input'] = df['input'].apply(json.loads)
df['output'] = df['output'].apply(json.loads)
# Model eğitimi için input-output eşlerini tek bir string halinde düzenle
df["text"] = df.apply(lambda row: f"### Girdi:\n{json.dumps(row['input'], ensure_ascii=False)}\n\n### Cevap:\n{json.dumps(row['output'], ensure_ascii=False)}", axis=1)

dataset = Dataset.from_pandas(df[["text"]])



if "COLAB_" not in "".join(os.environ.keys()):
     !pip install unsloth
     sudo apt update
     sudo apt install libcurl4-openssl-dev

df = pd.read_csv("fine_tune_data.csv")

df['text'] = df['input'].astype(str) + " ### " + df['output'].astype(str)
from datasets import Dataset

# HuggingFace Dataset formatına çevir
dataset = Dataset.from_pandas(df[['text']])

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4-mini-128k-instruct",
    max_seq_length=2048,
    dtype="auto",
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
def format_example(example):
    return {
        "text": f"{example['text']}"
    }

dataset = dataset.map(format_example)

tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=2048),
    batched=True
)
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="phi4-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=args,
    tokenizer=tokenizer,
)

model.config.use_cache = False
trainer.train()

model.save_pretrained("finetuned_phi4_lora")
tokenizer.save_pretrained("finetuned_phi4_lora")

"""
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import torch
from unsloth 
import FastLanguageModel


dataset = load_dataset("csv", data_files="")["train"]
dataset = dataset.select(columns=["input", "output"])

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-4",
    max_seq_length=256,
    dtype=torch.float16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    #task_type="CAUSAL_LM",
)
"""def format_prompt(sample):
    i = sample["input_"]
    o = sample["output_"]
    return {"text": f"input: {i}\noutput: {o}"}

dataset = dataset.map(lambda x: format_prompt(x))
tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=256),
    batched=True
)"""
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="phi-4",
)

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    convos = [
        [
            {"role": "user", "content": input_},
            {"role": "assistant", "content": output_}
        ]
        for input_, output_ in zip(inputs, outputs)
    ]

    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in convos
    ]

    return { "text": texts }

dataset = dataset.map(formatting_prompts_func, batched=True)
tokenized = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=256),
    batched=True
)

args = TrainingArguments(
    output_dir="lora_qawiki",
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    max_steps=50,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="no",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized,
    args=args
)

model.config.use_cache = False
trainer.train()
model.save_pretrained("model_06")


"""def answer_question(question):
    prompt = f"Soru: {question}\nCevap:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip()

print(answer_question("hello, whatsapp?"))"""


