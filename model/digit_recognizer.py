from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization

def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten the input images
        Dense(512, activation='relu'),  # Hidden layer with 512 neurons
        Dropout(0.3),  # Dropout layer to prevent overfitting
        Dense(256, activation='relu'),  # Hidden layer with 256 neurons
        Dropout(0.3),  # Dropout layer
        Dense(128, activation='relu'),  # Hidden layer with 128 neurons
        BatchNormalization(),  # Batch normalization layer for faster convergence
        Dense(10, activation='softmax')  # Output layer with 10 neurons (for 10 digits)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def predict_digit(model, image):
    image = image.reshape(1, 28, 28)
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    return digit
