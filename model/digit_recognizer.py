import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Load datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Separate features and labels
X_train = train_data.drop('label', axis=1) / 255.0
y_train = train_data['label']

# Reshape input data to match the expected shape
X_train = X_train.values.reshape(-1, 28, 28)

# Define the model
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten the input images
        Dense(128, activation='relu'),  # Hidden layer with 128 neurons
        Dense(10, activation='softmax')  # Output layer with 10 neurons (for 10 digits)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, epochs=10):
    X_train_data, X_val, y_train_data, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    history = model.fit(X_train_data, y_train_data, epochs=epochs, validation_data=(X_val, y_val))
    return history

# Predict digits
def predict_digit(model, image):
    image = image.reshape(1, 28, 28)
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    return digit
