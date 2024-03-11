from flask import Flask, render_template, request
from model.digit_recognizer import build_model, predict_digit
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            model = build_model()  # Build the digit recognition model
            digit = predict_digit(model, filename)  # Predict the digit
            return render_template('index.html', digit=digit)

if __name__ == '__main__':
    app.run(debug=True)