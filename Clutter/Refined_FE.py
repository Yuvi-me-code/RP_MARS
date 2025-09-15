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
    """ Extract features from an image """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Unable to read image '{image_path}'. Skipping...")
        return None
    try:
        msgggf = extract_msgggf(image)
        msesgf = extract_msesgf(image)
        msfdmaf = extract_msfdmaf(image)
        msssf = extract_msssf(image)
        mssamf = extract_mssamf(image)
        return np.concatenate((msgggf, msesgf, msfdmaf, msssf, mssamf))
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_images(parent_folder, output_csv):
    """ Process images from class folders and save extracted features """
    feature_list = []
    
    for class_folder in ["Class_0", "Class_1", "Class_2"]:
        class_path = os.path.join(parent_folder, class_folder)
        if not os.path.exists(class_path):
            print(f"Warning: Folder '{class_path}' not found. Skipping...")
            continue
        
        class_label = int(class_folder.split('_')[-1])
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        if not image_files:
            print(f"Warning: No images found in '{class_path}'. Skipping...")
            continue

        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            features = extract_features(image_path)
            if features is not None:
                feature_list.append([image_file, class_label] + features.tolist())
    
    if not feature_list:
        print("Error: No features extracted. Check image files and formats.")
        return

    columns = ['Image_Name', 'Class_Label'] + [f'MSGGGF_{i}' for i in range(10)] + ['MSESGF'] + ['MSFDMAF'] + ['MSSSF'] + ['MSSAMF']
    df = pd.DataFrame(feature_list, columns=columns)
    df.to_csv(output_csv, index=False)
    
    print(f"Feature extraction completed. Output saved to {output_csv}")

def main():
    parent_folder = r"Data\Final"
    output_csv = "final_features.csv"
    process_images(parent_folder, output_csv)

if __name__ == "__main__":
    main()
