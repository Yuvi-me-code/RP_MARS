import os
import cv2
import numpy as np
import pickle
import pandas as pd
from skimage.feature import canny
from scipy.fftpack import fft2, fftshift
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
model_filename = r"final_gabor_best_model.pkl"
with open(model_filename, 'rb') as model_file:
    trained_rf = pickle.load(model_file)
print("Model loaded successfully.")

# Feature extraction functions
def extract_msgggf(image_gray):
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.histogram(gradient_magnitude, bins=10, range=(0, 255))[0]

def extract_msesgf(image_gray):
    edges = canny(image_gray, sigma=1.0)
    return np.array([np.sum(edges)])

def extract_msfdmaf(image_gray):
    f_transform = fft2(image_gray)
    f_transform_shifted = fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    return [np.mean(magnitude_spectrum)]

def extract_msssf(image_gray):
    f_transform = fft2(image_gray)
    f_transform_shifted = fftshift(f_transform)
    spectrum = np.abs(f_transform_shifted)
    center = spectrum.shape[0] // 2
    left = np.sum(spectrum[:, :center])
    right = np.sum(spectrum[:, center:])
    return [np.abs(left - right)]

def extract_mssamf(image_gray):
    f_transform = fft2(image_gray)
    f_transform_shifted = fftshift(f_transform)
    spectrum = np.abs(f_transform_shifted)
    center = spectrum.shape[0] // 2, spectrum.shape[1] // 2
    distances = np.sqrt((np.arange(spectrum.shape[0])[:, None] - center[0])**2 +
                        (np.arange(spectrum.shape[1]) - center[1])**2)
    return [np.sum(spectrum * distances)]

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Unable to read image '{image_path}'.")
        return None
    msgggf = extract_msgggf(image)
    msesgf = extract_msesgf(image)
    msfdmaf = extract_msfdmaf(image)
    msssf = extract_msssf(image)
    mssamf = extract_mssamf(image)
    return np.concatenate((msgggf, msesgf, msfdmaf, msssf, mssamf))

def predict_image_class(image_path):
    features = extract_features(image_path)
    if features is None:
        print("Error: Feature extraction failed.")
        return
    
    features = features.reshape(1, -1)
    predicted_class = trained_rf.predict(features)[0]
    class_mapping = {0: "Rocky", 1: "Hard", 2: "Sandy"}
    print(f"Predicted Class: {class_mapping.get(predicted_class, 'Unknown')}")
    return class_mapping.get(predicted_class, "Unknown")

image_path = r"WhatsApp Image 2025-06-01 at 17.47.50_b305b76f.jpg"
predict_image_class(image_path)