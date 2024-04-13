from flask import Flask, request, jsonify, redirect , render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_path = "./mbti_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
id2label = model.config.id2label


@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.json['sentence']
    print(sentence)
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_label_idx = torch.argmax(logits, axis=1).item()
    predicted_label = id2label[predicted_label_idx].lower()  # Ensure labels are formatted correctly
    return jsonify({'type': predicted_label})

@app.route('/redirect/<mbti_type>')
def redirect_to_profile(mbti_type):
    url = f"https://www.16personalities.com/{mbti_type}-personality"
    return redirect(url)

if __name__ == "__main__":
    app.run(debug=True)