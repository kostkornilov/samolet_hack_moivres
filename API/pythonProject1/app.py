from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import os
import json
from safetensors.torch import load_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_dir = os.path.join(os.path.dirname(__file__), "model")
tokenizer_dir = os.path.join(os.path.dirname(__file__), "tokenizer")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
pipe = pipeline('ner', model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    return "Server is running"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        text = input_data['text']
        result = pipe(text)
        for prediction in result:
            prediction['score'] = round(float(prediction['score']), 4)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
