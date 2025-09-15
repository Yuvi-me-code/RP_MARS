import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import gabor
from skimage.feature import canny
from scipy.stats import skew, kurtosis

# Define Gabor filter parameters
def apply_gabor_filters(image_gray, frequencies=[0.1, 0.3, 0.5], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    features = []
    for freq in frequencies:
        for theta in thetas:
            real, _ = gabor(image_gray, frequency=freq, theta=theta)
            features.append(np.mean(real))   # Mean response
            features.append(np.var(real))    # Variance response
            features.append(np.sum(real**2)) # Energy
    return features

# Edge detection feature
def extract_canny_edges(image_gray):
    edges = canny(image_gray, sigma=1.0)
    return np.array([np.sum(edges)])  # Sum of edges

# Compute histogram-based features
def extract_histogram_features(image_gray, bins=10):
    hist, _ = np.histogram(image_gray, bins=bins, range=(0, 255))
    return hist

# Extract statistical features
def extract_statistical_features(image_gray):
    return [np.mean(image_gray), np.var(image_gray), skew(image_gray.flatten()), kurtosis(image_gray.flatten())]

# Feature extraction pipeline
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Unable to read image '{image_path}'. Skipping...")
        return None
    try:
        gabor_features = apply_gabor_filters(image)
        edge_features = extract_canny_edges(image)
        hist_features = extract_histogram_features(image)
        stat_features = extract_statistical_features(image)
        return np.concatenate((gabor_features, edge_features, hist_features, stat_features))
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Process images and save features
def process_images(parent_folder, output_csv):
    feature_list = []
    
    for class_folder in ["Class_0", "Class_1", "Class_2"]:
        class_path = os.path.join(parent_folder, class_folder)
        if not os.path.exists(class_path):
            print(f"Warning: Folder '{class_path}' not found. Skipping...")
            continue
        
        class_label = int(class_folder.split('_')[-1])
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            features = extract_features(image_path)
            if features is not None:
                feature_list.append([image_file, class_label] + features.tolist())
    
    if not feature_list:
        print("Error: No features extracted. Check image files and formats.")
        return

    # Define feature column names
    columns = ['Image_Name', 'Class_Label'] + [f'Gabor_{i}' for i in range(36)] + ['Canny_Edges'] + \
              [f'Hist_{i}' for i in range(10)] + ['Mean', 'Variance', 'Skewness', 'Kurtosis']
    
    df = pd.DataFrame(feature_list, columns=columns)
    df.to_csv(output_csv, index=False)
    
    print(f"Feature extraction completed. Output saved to {output_csv}")

# Main function to execute feature extraction
def main():
    parent_folder = r"Final\Final"
    output_csv = "mars_gabor_features.csv"
    process_images(parent_folder, output_csv)

if __name__ == "__main__":
    main()
