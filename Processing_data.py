
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}