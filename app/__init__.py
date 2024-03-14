from flask import Flask, request, render_template
import os
import sys
import numpy as np
from PIL import Image

# Append the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from digit_recognizer import build_model, predict_digit

app = Flask(__name__)

# Load your model here
model = build_model()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream).convert('L')  # Convert to grayscale
            image = image.resize((28, 28)) 
            image = np.array(image) / 255.0 

            digit = predict_digit(model, image)
            return render_template('index.html', digit=digit)
    return render_template('index.html')
