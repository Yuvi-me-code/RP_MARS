import pandas as pd

# Load the CSV file
df = pd.read_csv('mars_gabor_features.csv')  # Replace with your actual file path

# Extract feature names
feature_names = list(df.columns)

# Save feature names to a text file
with open('feature_names.txt', 'w') as f:
    for feature in feature_names:
        f.write(f"{feature}\n")

print("Feature names saved to feature_names.txt")
