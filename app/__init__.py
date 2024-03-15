# Import necessary modules
from flask import Flask, request, render_template  # Import Flask for creating web application, request for handling HTTP requests, and render_template for rendering HTML templates
import os  # Import os module for interacting with the operating system
import sys  # Import sys module for accessing system-specific parameters and functions
import numpy as np  # Import numpy for numerical operations
from PIL import Image  # Import Image module from PIL library for image processing

# Append the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))  # Add the parent directory path to the system path to access the model directory

# Import functions from the digit_recognizer module
from digit_recognizer import build_model, predict_digit  # Import functions build_model and predict_digit from digit_recognizer module

# Create a Flask application instance
app = Flask(__name__)

# Load the trained model
model = build_model()  # Call the build_model function to load the trained model

# Define a route for handling file uploads
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':  # Check if the request method is POST
        file = request.files['file']  # Get the uploaded file from the request
        if file:  # Check if a file was uploaded
            # Open the uploaded image file, convert it to grayscale, resize it to 28x28 pixels, and normalize the pixel values
            image = Image.open(file.stream).convert('L').resize((28, 28))  # Convert image to grayscale and resize to 28x28 pixels
            image = np.array(image) / 255.0  # Normalize pixel values to range [0, 1]

            # Use the trained model to predict the digit from the uploaded image
            digit = predict_digit(model, image)  # Call the predict_digit function to predict the digit from the image
            return render_template('index.html', digit=digit)  # Render the HTML template with the predicted digit
    return render_template('index.html')  # Render the HTML template for uploading files

# To run the Flask application
if __name__ == '__main__':
    app.run()
