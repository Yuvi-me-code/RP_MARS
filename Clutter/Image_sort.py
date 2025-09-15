import os
import shutil
import pandas as pd

# Load the classified CSV file
csv_file = "classified_output_features.csv"
df = pd.read_csv(csv_file)

# Define paths
original_image_folder = r"C:\Users\Yuvraj\OneDrive\Files\Work\MARS\msl-images\calibrated"
destination_folder = r"C:\Users\Yuvraj\OneDrive\Files\Work\MARS\sorted_images"

# Create subfolders for each class
for class_label in [0, 1, 2]:
    class_folder = os.path.join(destination_folder, f"Class_{class_label}")
    os.makedirs(class_folder, exist_ok=True)

# Copy images into their respective class folders
for _, row in df.iterrows():
    image_name = row['Image_Name']
    class_label = row['Class']
    
    src_path = os.path.join(original_image_folder, image_name)
    dest_path = os.path.join(destination_folder, f"Class_{class_label}", image_name)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)
    else:
        print(f"Warning: {image_name} not found in {original_image_folder}")

print("Images sorted and copied successfully!")
