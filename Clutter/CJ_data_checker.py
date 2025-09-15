import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model_filename = r"Clutter\best_model.pkl"
with open(model_filename, 'rb') as model_file:
    trained_model = pickle.load(model_file)
print("Model loaded successfully.")

# Load new dataset for evaluation
csv_file = r"CSVs\FINAL_RESULT.csv"  # Change to actual file path
df = pd.read_csv(csv_file)

# Check column names
print("Columns in dataset:", df.columns)

# Extract features and labels
X_new = df.iloc[:, 1:-1].values  # Ensure correct slicing
y_new = df.iloc[:, -1].values  # Labels are in the last column

# Check feature shape
print("Expected feature count:", trained_model.n_features_in_)
print("Feature shape in test data:", X_new.shape)

# Manually encode labels: 'rocky' -> 0, 'hard' -> 1, 'sandy' -> 2
label_mapping = {"rocky": 0, "hard": 1, "sandy": 2}
y_new = np.array([label_mapping[label] for label in y_new])

# Predict using the loaded model
print("Processing new dataset...")
y_pred_new = trained_model.predict(X_new)

# Evaluate accuracy
accuracy_new = accuracy_score(y_new, y_pred_new)
print(f"Model Accuracy on New Data: {accuracy_new * 100:.2f}%")
print("Classification Report:\n", classification_report(y_new, y_pred_new))
