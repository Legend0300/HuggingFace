from transformers import AutoTokenizer
import torch

from transformers import AutoModelForSequenceClassification


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

print(outputs.last_hidden_state.shape)

print(outputs.logits.shape)
print(outputs.logits)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

print(model.config.id2label)
