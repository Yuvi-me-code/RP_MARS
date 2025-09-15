import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Read the CSV file
# Replace 'your_file.csv' with your actual file path
df = pd.read_csv('mars_gabor_features.csv')

# Step 2: Separate features (optional: remove target column if present)
# If your dataset has a target column (e.g., 'target'), remove it from scaling
# Uncomment and modify the line below if needed
target = df['Class_Label'] 
image = df['Image_Name']
features = df.drop(columns=['Class_Label', 'Image_Name'], errors='ignore')

# For now, we'll scale the entire dataset
# features = df

# Step 3: Apply StandardScaler
scaler = StandardScaler()
scaled_array = scaler.fit_transform(features)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Step 4: Convert scaled data back to DataFrame
scaled_df = pd.DataFrame(scaled_array, columns=features.columns)

# Optional: Add target column back
scaled_df['Class_Label'] = target
scaled_df['Image_Name'] = image

# Step 5: Save to new CSV (optional)
scaled_df.to_csv('scaled_output.csv', index=False)

# Show the first few rows
print(scaled_df.head())
print(scaled_df.size())
