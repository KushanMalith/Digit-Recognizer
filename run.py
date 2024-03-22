import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

# Load datasets
train_data = pd.read_csv('data/train.csv')

# Separate features and labels
y_train = train_data['label']
X_train = train_data.drop(columns=['label'], axis=1) / 255.0  # Normalize pixel values

# Reshape input data to match the expected shape
X_train = X_train.values.reshape(-1, 28, 28, 1)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    Flatten(),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint_path = 'model/digit_recognizer_model.h5'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=False)

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[checkpoint_callback])

# Save the trained model
model.save(checkpoint_path)
