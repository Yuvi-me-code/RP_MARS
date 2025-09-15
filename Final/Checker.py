import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model_filename = r"final_gabor_best_model.pkl"
with open(model_filename, 'rb') as model_file:
    trained_rf = pickle.load(model_file)
print("Model loaded successfully.")

# Load new dataset for testing
new_csv_file = r"mars_gabor_features.csv"
df_new = pd.read_csv(new_csv_file)

# Check if the label column exists
if "Class_Label" not in df_new.columns:
    print("Error: 'Class_Label' column not found in the dataset!")
    exit()

# Extract features and labels
X_new = df_new.iloc[:, 2:].values  # Feature columns (excluding image name and label column)
y_new = df_new["Class_Label"].values  # Ensure correct label selection

# Debugging: Check unique labels
print("Unique labels in training:", np.unique(trained_rf.classes_))
print("Unique labels in test data:", np.unique(y_new))

# Predict using the loaded model
print("Processing new dataset...")
y_pred_new = trained_rf.predict(X_new)

# Evaluate accuracy
accuracy_new = accuracy_score(y_new, y_pred_new)
print(f"Random Forest Accuracy on New Data: {accuracy_new * 100:.2f}%")
print("Classification Report:\n", classification_report(y_new, y_pred_new, zero_division=1))
