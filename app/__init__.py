# Import necessary modules
from flask import Flask, request, render_template
import os
import sys
import numpy as np
from PIL import Image

# Append the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

# Import functions from the digit_recognizer_predict module
from digit_recognizer_predict import predict_digit

# Create a Flask application instance
app = Flask(__name__)

# Configure the static folder
app.static_folder = 'static'

# Define a route for handling file uploads
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream)
            image_data = np.array(image).reshape(-1, 28, 28, 1)
            digit = predict_digit(image_data)
            return render_template('index.html', digit=digit)
    return render_template('index.html')

# To run the Flask application
if __name__ == '__main__':
    app.run()
