import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import canny
from scipy.fftpack import fft2, fftshift

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
        print(f"Warning: Unable to read image '{image_path}'. Skipping...")
        return np.concatenate(([os.path.basename(image_path)], np.zeros(14)))  # Return zeros if image can't be read
    msgggf = extract_msgggf(image)
    msesgf = extract_msesgf(image)
    msfdmaf = extract_msfdmaf(image)
    msssf = extract_msssf(image)
    mssamf = extract_mssamf(image)
    return np.concatenate(([os.path.basename(image_path)], msgggf, msesgf, msfdmaf, msssf, mssamf))

def process_images(folder_path, output_csv):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not image_files:
        print(f"Error: No images found in '{folder_path}'.")
        return

    print(f"Processing {len(image_files)} images from '{folder_path}'...")

    feature_list = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing: {image_path}")  # Debugging
        features = extract_features(image_path)
        feature_list.append(features)
    
    columns = ['Image_Name'] + [f'MSGGGF_{i}' for i in range(10)] + ['MSESGF'] + ['MSFDMAF'] + ['MSSSF'] + ['MSSAMF']
    
    df = pd.DataFrame(feature_list, columns=columns)
    df.to_csv(output_csv, index=False)
    
    print(f"Feature extraction completed. Output saved to {output_csv}")

def main():
    folder_path = r"C:\Users\Yuvraj\OneDrive\Files\Work\MARS\msl-images\calibrated"
    output_csv = "output_features.csv"
    process_images(folder_path, output_csv)

if __name__ == "__main__":
    main()
