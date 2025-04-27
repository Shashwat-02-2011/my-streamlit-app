# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# Step 1: Load the dataset
df = pd.read_csv('heart.csv')  # Make sure heart.csv is in your project folder

# Step 2: Split data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Step 3: Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale features (Important for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and train the SVM model
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Step 6: Evaluate the model
accuracy = svm_model.score(X_test_scaled, y_test)
print(f'Model Accuracy: {accuracy:.2f}')

# Step 7: Save the model and scaler
with open('heart_disease_svm_model.sav', 'wb') as model_file:
    pickle.dump(svm_model, model_file)

with open('scaler.sav', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model and scaler saved successfully!")
