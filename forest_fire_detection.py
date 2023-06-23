# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset (replace 'dataset.csv' with your actual dataset file)
dataset = pd.read_csv('dataset.csv')

# Split the dataset into features (X) and labels (y)
X = dataset.drop('target_variable', axis=1)  # Adjust 'target_variable' with the actual target column name
y = dataset['target_variable']  # Adjust 'target_variable' with the actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier
clf = RandomForestClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
