import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
csv_file = r"Final\combined_features.csv"
df = pd.read_csv(csv_file)

# Extract features and labels
X = df.iloc[:, 2:].values  # All feature columns (excluding image name and label)
y = df["Class_Label"].values  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define RandomForest model and hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model
tuned_rf = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Save the trained model
model_filename = "random_forest_model.pkl"
with open(model_filename, 'wb') as model_file:
    pickle.dump(tuned_rf, model_file)
print(f"Model saved as {model_filename}")

# Evaluate the tuned model
y_pred = tuned_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
