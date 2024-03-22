import os
import sys
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model/digit_recognizer_model.keras')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from request
    img_data = request.files['file']
    # Process image data
    img_arr = process_image(img_data)
    # Make prediction
    prediction = model.predict_classes(img_arr)
    return jsonify({'prediction': str(prediction[0])})

# Function to process image data
def process_image(img_data):
    # Convert image to grayscale and resize to 28x28
    img = Image.open(img_data).convert('L').resize((28, 28))
    # Convert image to numpy array and normalize pixel values
    img_arr = np.array(img) / 255.0
    # Reshape image array to match model input shape
    img_arr = img_arr.reshape(-1, 28, 28, 1)
    return img_arr

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
