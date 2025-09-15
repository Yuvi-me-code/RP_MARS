import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load extracted features from CSV
csv_file = "output_features.csv"
df = pd.read_csv(csv_file)

# Extract feature columns (excluding image names)
feature_columns = df.columns[1:]
X = df[feature_columns].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering to classify into 3 terrain types
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Class'] = kmeans.fit_predict(X_scaled)

# Apply KNN for refining classification
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, df['Class'])
predicted_classes = knn.predict(X_scaled)

# Improve clustering by refining labels using KNN
accuracy = accuracy_score(df['Class'], predicted_classes)
print(f"Initial Clustering Accuracy: {accuracy * 100:.2f}%")

df['Class'] = predicted_classes  # Update class labels with refined predictions

# Save updated CSV with class labels
df.to_csv("classified_output_features.csv", index=False)
print("Classification completed and saved to 'classified_output_features.csv'")
