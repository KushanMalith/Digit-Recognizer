import os
import sys
import pandas as pd
from model.digit_recognizer_train import build_model

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Separate features and labels
y_train = train_data['pixel0']  # Use 'pixel0' column as labels
X_train = train_data.drop(columns=['pixel0'], axis=1) / 255.0  # Exclude 'pixel0' column and normalize pixel values

# Reshape input data to match the expected shape
X_train = X_train.values.reshape(-1, 28, 28, 1)

# Build and train the model
model = build_model()
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Save the trained model
model.save('model/digit_recognizer_model.h5')
