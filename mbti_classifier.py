from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
from datasets import load_dataset, DatasetDict
import evaluate
import torch

# Load your datasets
raw_datasets = load_dataset("Legend0300/MBTI")

# Perform a train-test split
train_test_split = raw_datasets['train'].train_test_split(test_size=0.1)  # 10% for testing
datasets = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Define unique labels and create mapping dictionaries
def find_unique_labels(dataset, label_column):
    labels = dataset[label_column]
    unique_labels = list(set(labels))
    return unique_labels

def create_label_mappings(unique_labels):
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label

unique_labels = find_unique_labels(datasets["train"], 'Type')
label2id, id2label = create_label_mappings(unique_labels)

# Prepare the tokenizer and model
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize(examples):
    tokenized_inputs = tokenizer(examples["Sentence"], padding="max_length", truncation=True)
    tokenized_inputs["labels"] = [label2id[label] for label in examples["Type"]]
    return tokenized_inputs

# Tokenize the data
tokenized_datasets = datasets.map(tokenize, batched=True)

# Load the accuracy metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(unique_labels)
)

model.config.id2label = id2label

# Training arguments
args = TrainingArguments(
    "MBTI-Classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=15,
    weight_decay=0.01,
    per_device_train_batch_size=4,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

def predict_mbti(sentence, tokenizer, model, label2id, id2label):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True)

    # Move the inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to label index
    predicted_label_idx = torch.argmax(logits, axis=1).item()

    # Map the predicted label index to the label name
    predicted_label = id2label[predicted_label_idx]
    return predicted_label

model.push_to_hub("nielsr/my-awesome-bert-model")


# Save the model and tokenizer
model_path = "./mbti_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)


# Example of using the predict function
# sentence = "driven by a deep desire to strategize and improve systems and processes."
# predicted_type = predict_mbti(sentence, tokenizer, model, label2id, id2label)
# print(f"The predicted MBTI type for the sentence is: {predicted_type}")

