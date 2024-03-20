import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping

# Load datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Separate features and labels
y_train = train_data['pixel0']  # Use 'pixel0' column as labels
X_train = train_data.drop(columns=['pixel0'], axis=1) / 255.0  # Exclude 'pixel0' column and normalize pixel values

# Reshape input data to match the expected shape
X_train = X_train.values.reshape(-1, 28, 28, 1)

# Define the model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=50):
    X_train_data, X_val, y_train_data, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  
    history = model.fit(X_train_data, y_train_data, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return history

# Predict digits
def predict_digit(model, image):
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    return digit

# Build and train the model
model = build_model()
history = train_model(model, X_train, y_train)

# Function to reset model state
def reset_model():
    global model
    model = build_model()

# Function to make prediction
def make_prediction(image):
    reset_model()  # Reset model for each prediction
    return predict_digit(model, image)
