import numpy as np
from flask import Flask, request, render_template
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import pickle

app = Flask(__name__)

model = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index5.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Image, question
    im_path = Image.open("templates/testimage2.jpg").convert("RGB")
    req = [str(x) for x in request.form.values()]

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    encodings = processor(im_path, req[0], return_tensors="pt").to("cpu")
    outputs = model(**encodings)
    logits = outputs.logits
    _, answer_index_top5 = torch.topk(logits, 5)

    predicted_answer = []
    for pred_answer_index in answer_index_top5[0, :]:
        predicted_answer.append(model.config.id2label[pred_answer_index.item()])

    return render_template('index5.html', prediction_text='Predicted output: {}'.format(predicted_answer))

if __name__ == "__main__":
    app.debug = True
    app.run()