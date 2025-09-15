import os
import cv2
import numpy as np
import pandas as pd
import pickle
import base64
import requests
import time
from flask import Flask, request, jsonify
from skimage.feature import canny
from scipy.fftpack import fft2, fftshift
from scipy.stats import skew, kurtosis

# Load trained model
MODEL_PATH = r"New_Feature_Extraction\Clutter\best_model.pkl"
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

def extract_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    msgggf = np.histogram(gradient_magnitude, bins=10, range=(0, 255))[0]
    
    msesgf = np.array([np.sum(canny(image_gray, sigma=1.0))])
    
    f_transform = fft2(image_gray)
    f_transform_shifted = fftshift(f_transform)
    spectrum = np.abs(f_transform_shifted)
    msfdmaf = [np.mean(spectrum)]
    msssf = [np.abs(np.sum(spectrum[:, :spectrum.shape[1]//2]) - np.sum(spectrum[:, spectrum.shape[1]//2:]))]
    mssamf = [np.sum(spectrum * np.arange(spectrum.shape[0])[:, None])]
    
    fft_spectrum = cv2.dft(np.float32(image_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_shifted = np.fft.fftshift(fft_spectrum)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(fft_shifted[:, :, 0], fft_shifted[:, :, 1]) + 1)
    
    fdma_feature = [np.mean(magnitude_spectrum)]
    amp_moments = [np.mean(magnitude_spectrum), np.var(magnitude_spectrum), skew(magnitude_spectrum.flatten()), kurtosis(magnitude_spectrum.flatten())]
    
    sym_x, sym_y = np.mean(np.abs(magnitude_spectrum[:magnitude_spectrum.shape[0]//2, :] - np.flipud(magnitude_spectrum[magnitude_spectrum.shape[0]//2:, :]))), np.mean(np.abs(magnitude_spectrum[:, :magnitude_spectrum.shape[1]//2] - np.fliplr(magnitude_spectrum[:, magnitude_spectrum.shape[1]//2:])))
    
    return np.concatenate((msgggf, msesgf, msfdmaf, msssf, mssamf, fdma_feature, amp_moments, [sym_x, sym_y]))
def process_predictions():
    while True:
        response = requests.get("http://localhost:5000/check_area").json()
        if response.get("check"):
            image_data = requests.get("http://localhost:5000/latest_image").json()
            image_b64 = image_data["image"]
            coordinates = image_data["coordinates"]
            
            image_bytes = base64.b64decode(image_b64)
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            
            features = extract_features(image).reshape(1, -1)
            prediction = model.predict(features)[0]
            accuracy = max(model.predict_proba(features)[0])
            
            label_map = {0: "rocky_terrain", 1: "hard_terrain", 2: "sandy_terrain"}
            result = {
                "status": "success",
                "class": label_map[prediction],
                "accuracy": f"{accuracy:.2f}",
                "coordinate": coordinates
            }
            
            requests.post("http://localhost:5000/predicted_area", json=result)
        time.sleep(2)

if __name__ == "__main__":
    process_predictions()
