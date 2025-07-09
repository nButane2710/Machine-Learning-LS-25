from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import numpy as np


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
# print(dataset["train"][0])

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_column=["text"])

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir="./Week-4/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    save_total_limits=2,
)

trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()
perplexity = np.exp(eval_results["eval_loss"])
print(f"Perplexity: {perplexity}")

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

# Make sure we're on the same device as the model (CPU or GPU)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate continuation (predict next few tokens)
outputs = model.generate(
    **inputs,
    max_new_tokens=5,        # predict next 5 tokens
    do_sample=True,          # sample randomly (more creative)
    top_k=50,                # consider only top 50 tokens
    temperature=0.8          # randomness: lower = more conservative
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
