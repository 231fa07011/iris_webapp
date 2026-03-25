# train.py
# This script trains a machine learning model and saves it to a file

# Import necessary libraries
# --------------------------
# pandas: for data manipulation
# sklearn: for machine learning algorithms
# joblib: for saving/loading models
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Step 1.1: Load the Iris dataset
# -------------------------------
# The Iris dataset is built into scikit-learn
# It contains 150 samples of iris flowers with 4 features each
print("Loading Iris dataset...")
iris = load_iris()

# The dataset comes as a dictionary-like object
# - data: the measurements (features)
# - target: the species (0, 1, or 2)
# - target_names: the actual species names
# - feature_names: names of the measurements
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)

# Let's see what the data looks like
print("\nFirst 5 rows of data:")
print(pd.DataFrame(X, columns=iris.feature_names).head())
print("\nCorresponding species (0,1,2):", y[:5])
print("Species names:", iris.target_names)

# Step 1.2: Split data into training and testing sets
# ---------------------------------------------------
# train_test_split randomly splits the data
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 ensures we get the same split every time (for reproducibility)
print("\nSplitting data into training (80%) and testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # This ensures the split has same proportion of species
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 1.3: Create and train the model
# ------------------------------------
# Random Forest is an ensemble of decision trees
# n_estimators=100 means we create 100 trees
# random_state=42 for reproducibility
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # This is where the learning happens!

# Step 1.4: Evaluate the model
# ----------------------------
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy on test set: {accuracy * 100:.2f}%")

# Let's see which species are confused with each other
print("\nPredictions vs Actual (first 10 test samples):")
for i in range(10):
    pred_species = iris.target_names[y_pred[i]]
    actual_species = iris.target_names[y_test[i]]
    print(f"Sample {i+1}: Predicted={pred_species}, Actual={actual_species}")

# Step 1.5: Save the trained model to a file
# -----------------------------------------
# joblib.dump() serializes the model and saves it
# This file will be loaded by our web app
model_filename = 'iris_model.joblib'
joblib.dump(model, model_filename)
print(f"\n✅ Model saved as '{model_filename}'")

# Also save the feature names and target names for later use
model_info = {
    'feature_names': iris.feature_names,
    'target_names': iris.target_names
}
joblib.dump(model_info, 'model_info.joblib')
print("✅ Model info saved as 'model_info.joblib'")

# Step 1.6: Quick test - make a prediction with sample values
# ---------------------------------------------------------
print("\n🔍 Testing with a sample flower:")
# Let's use the first flower from the dataset
sample_features = X[0]  # This is a setosa
print(f"Features: {dict(zip(iris.feature_names, sample_features))}")

# Make prediction
prediction = model.predict([sample_features])[0]
probability = model.predict_proba([sample_features])[0]

print(f"Predicted species: {iris.target_names[prediction]}")
print(f"Confidence scores:")
for i, prob in enumerate(probability):
    print(f"  - {iris.target_names[i]}: {prob*100:.1f}%")