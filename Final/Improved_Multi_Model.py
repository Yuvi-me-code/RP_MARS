import pandas as pd
import numpy as np
import pickle
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import torch

# Check for GPU availability
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

device = get_device()
print(f"Using device: {device}")

# Load dataset
csv_file = r"mars_gabor_features.csv"
df = pd.read_csv(csv_file)

# Extract features and labels
X = df.iloc[:, 2:].values  # Features (excluding image name and label)
y = df["Class_Label"].values  # Labels
feature_names = df.columns[2:]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Selection using Recursive Feature Elimination with Cross-Validation
rf = GradientBoostingClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy', n_jobs=-1)
rfecv.fit(X_train, y_train)

# Select best features
X_train_selected = rfecv.transform(X_train)
X_test_selected = rfecv.transform(X_test)
selected_features = feature_names[rfecv.support_]
print("Selected Features:", selected_features)

# Define hyperparameter tuning using Optuna
def objective(trial):
    model_name = trial.suggest_categorical("model", ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting"])
    
    if model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_uniform("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            tree_method='gpu_hist' if device == "cuda" else "auto",
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42
        )
    elif model_name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_uniform("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            device="gpu" if device == "cuda" else "cpu",
            random_state=42
        )
    elif model_name == "CatBoost":
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 500),
            depth=trial.suggest_int("depth", 5, 10),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            bootstrap_type="Bernoulli",  # Ensure compatibility with GPU
            task_type="GPU" if device == "cuda" else "CPU",
            grow_policy="Depthwise",  # Safer alternative to SymmetricTree
            verbose=0,
            random_state=42
        )
    else:  # Gradient Boosting
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_uniform("subsample", 0.5, 1.0),
            random_state=42
        )
    
    score = cross_val_score(model, X_train_selected, y_train, cv=5, scoring="accuracy").mean()
    return score

# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Train the best model
best_params = study.best_params
best_model_name = best_params.pop("model")

print(f"Best Model: {best_model_name}")
print(f"Best Hyperparameters: {best_params}")

# Initialize the best model
if best_model_name == "XGBoost":
    best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="mlogloss", tree_method='gpu_hist' if device == "cuda" else "auto", random_state=42)
elif best_model_name == "LightGBM":
    best_model = LGBMClassifier(**best_params, device="gpu" if device == "cuda" else "cpu", random_state=42)
elif best_model_name == "CatBoost":
    best_model = CatBoostClassifier(**best_params, task_type="GPU" if device == "cuda" else "CPU", grow_policy="SymmetricTree", verbose=0, random_state=42)
else:
    best_model = GradientBoostingClassifier(**best_params, random_state=42)

# Train on the full training set
best_model.fit(X_train_selected, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the best model
model_filename = "improved_final_gabor_best_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(best_model, model_file)
print("Best model saved as 'improved_final_gabor_best_model.pkl'")