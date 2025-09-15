import cv2
import numpy as np
import pickle
import os
from skimage.filters import gabor
from skimage.feature import canny
from scipy.stats import skew, kurtosis
from PIL import Image

# ---------------------------
# Load Pre-trained Model
# ---------------------------
MODEL_PATH = "final_gabor_best_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Feature Extraction Functions
# ---------------------------
def apply_gabor_filters(image_gray, frequencies=[0.1, 0.3, 0.5], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    features = []
    for freq in frequencies:
        for theta in thetas:
            real, _ = gabor(image_gray, frequency=freq, theta=theta)
            features.append(np.mean(real))
            features.append(np.var(real))
            features.append(np.sum(real**2))
    return features

def extract_canny_edges(image_gray):
    edges = canny(image_gray, sigma=1.0)
    return [np.sum(edges)]

def extract_histogram_features(image_gray, bins=10):
    hist, _ = np.histogram(image_gray, bins=bins, range=(0, 255))
    return hist.tolist()

def extract_statistical_features(image_gray):
    return [
        np.mean(image_gray),
        np.var(image_gray),
        skew(image_gray.flatten()),
        kurtosis(image_gray.flatten())
    ]

def extract_features(image_gray):
    gabor_features = apply_gabor_filters(image_gray)
    edge_features = extract_canny_edges(image_gray)
    hist_features = extract_histogram_features(image_gray)
    stat_features = extract_statistical_features(image_gray)
    return np.concatenate((gabor_features, edge_features, hist_features, stat_features)).reshape(1, -1)

# ---------------------------
# Prediction Function
# ---------------------------
def predict_terrain(image_path):
    image = Image.open(image_path).convert("L")
    image_gray = np.array(image)
    features = extract_features(image_gray)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()

    terrain_map = {0: "rocky_terrain", 1: "hard_terrain", 2: "sandy_terrain"}
    terrain_class = terrain_map.get(prediction, "unknown")

    return terrain_class, probability

# ---------------------------
# Main Execution with Option
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mars Terrain Classification")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    args = parser.parse_args()

    if args.image:
        terrain, prob = predict_terrain(args.image)
        print(f"Image: {args.image}")
        print(f"Predicted Terrain: {terrain}")
        print(f"Confidence: {prob:.4f}\n")

    elif args.dir:
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        image_files = [f for f in os.listdir(args.dir) if os.path.splitext(f)[1].lower() in image_extensions]

        if not image_files:
            print("No valid image files found in the directory.")
        else:
            for image_file in image_files:
                image_path = os.path.join(args.dir, image_file)
                terrain, prob = predict_terrain(image_path)
                print(f"Image: {image_file}")
                print(f"Predicted Terrain: {terrain}")
                print(f"Confidence: {prob:.4f}\n")
    else:
        print("Please provide either --image <image_path> or --dir <directory_path>")
