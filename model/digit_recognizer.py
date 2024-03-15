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
X_train = train_data.drop('label', axis=1) / 255.0
y_train = train_data['label']

# Reshape input data to match the expected shape
X_train = X_train.values.reshape(-1, 28, 28, 1)

# Define the model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), padding='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=50):
    X_train_data, X_val, y_train_data, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  
    history = model.fit(X_train_data, y_train_data, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return history, X_val, y_val

# Predict digits
def predict_digit(model, image):
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    return digit

# Build and train the model
model = build_model()
history, X_val, y_val = train_model(model, X_train, y_train)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Accuracy:", accuracy)
