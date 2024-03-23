import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Separate features and labels
y_train = train_data['label']  # Use 'label' column as labels
X_train = train_data.drop(columns=['label'], axis=1) / 255.0  # Exclude 'label' column and normalize pixel values

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

# Train the model with augmented data
def train_model_with_augmentation(model, X_train, y_train, epochs=50):
    X_train_data, X_val, y_train_data, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create ImageDataGenerator with desired augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=10,  # Rotate the image by up to 10 degrees
        width_shift_range=0.1,  # Shift the image horizontally by up to 10% of the width
        height_shift_range=0.1,  # Shift the image vertically by up to 10% of the height
        zoom_range=0.1,  # Zoom into the image by up to 10%
        horizontal_flip=False,  # Randomly flip images horizontally
        vertical_flip=False  # Randomly flip images vertically
    )
    
    # Fit the ImageDataGenerator to X_train_data
    datagen.fit(X_train_data)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model using augmented data generator
    history = model.fit(datagen.flow(X_train_data, y_train_data, batch_size=32),
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping])
    
    return history

# Function to predict digit from image data
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



# Build and train the model with data augmentation
model = build_model()
history_with_augmentation = train_model_with_augmentation(model, X_train, y_train)

# Save the trained model
model.save('model/digit_recognizer_model.h5')
