import numpy as np
from tensorflow.keras.models import load_model

def predict_digit(image_data):
    # Load trained model
    model = load_model('model/digit_recognizer_model.h5')
    # Reshape image data within the function to match model input shape
    image_data = image_data.reshape(-1, 28, 28, 1)
    # Predict digit probabilities
    predictions = model.predict(image_data)
    # Get the index of the maximum probability as the predicted digit
    prediction = np.argmax(predictions[0])
    return prediction
