import pandas as pd                   # Pandas library for data manipulation
from sklearn.model_selection import train_test_split  # To import train_test_split function from scikit-learn for splitting data

# Loading datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Separating features and labels
X = train_data.drop('label', axis=1) / 255.0  # Features (normalize pixel values)
y = train_data['label']                      # Labels

# Spliting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshaping input data to match the expected shape
X_train = X_train.values.reshape(-1, 28, 28)  # Reshape training features
X_val = X_val.values.reshape(-1, 28, 28)      # Reshape validation features

# Printing the shape of training and validation sets
print("X_train shape:", X_train.shape)        # Print shape of training features
print("X_val shape:", X_val.shape)            # Print shape of validation features
print("y_train shape:", y_train.shape)        # Print shape of training labels
print("y_val shape:", y_val.shape)            # Print shape of validation labels
