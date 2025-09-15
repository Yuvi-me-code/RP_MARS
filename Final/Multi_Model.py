import pandas as pd
import numpy as np
import pickle
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load dataset
csv_file = r"mars_gabor_features.csv"
df = pd.read_csv(csv_file)

# Extract features and labels
X = df.iloc[:, 2:].values  # Features (excluding image name and label)
y = df["Class_Label"].values  # Labels

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter tuning using Optuna
def objective(trial):
    model_name = trial.suggest_categorical("model", ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "GradientBoosting"])
    
    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42
        )
    elif model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42
        )
    elif model_name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            random_state=42
        )
    elif model_name == "CatBoost":
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 50, 300),
            depth=trial.suggest_int("depth", 5, 10),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            verbose=0,
            random_state=42
        )
    else:  # Gradient Boosting
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            learning_rate=trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            random_state=42
        )
    
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score

# Run hyperparameter optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Train the best model
best_params = study.best_params
best_model_name = best_params.pop("model")

print(f"Best Model: {best_model_name}")
print(f"Best Hyperparameters: {best_params}")

# Initialize the best model
if best_model_name == "RandomForest":
    best_model = RandomForestClassifier(**best_params, random_state=42)
elif best_model_name == "XGBoost":
    best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
elif best_model_name == "LightGBM":
    best_model = LGBMClassifier(**best_params, random_state=42)
elif best_model_name == "CatBoost":
    best_model = CatBoostClassifier(**best_params, verbose=0, random_state=42)
else:
    best_model = GradientBoostingClassifier(**best_params, random_state=42)

# Train on the full training set
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the best model
model_filename = "final_gabor_best_model.pkl"
with open(model_filename, "wb") as model_file:
    pickle.dump(best_model, model_file)
print("Best model saved as 'best_model.pkl'")
