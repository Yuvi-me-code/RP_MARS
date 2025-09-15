import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shutil

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.CLAHE(clip_limit=2.0, p=0.2)
])

def augment_and_save(image_path, output_path, image_name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to read image '{image_path}'. Skipping...")
        return
    
    # Save original image
    shutil.copy(image_path, os.path.join(output_path, image_name))
    
    # Apply augmentations and save augmented images
    for i in range(50):  # Generate 3 augmented versions per image
        augmented = augmentation_pipeline(image=image)['image']
        augmented_name = f"aug_{i}_{image_name}"
        cv2.imwrite(os.path.join(output_path, augmented_name), augmented)

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Available folders:", os.listdir(input_folder))  # Debugging

    for class_folder in ["Class_0", "Class_1", "Class_2"]:
        class_path = os.path.join(input_folder, class_folder)
        output_class_path = os.path.join(output_folder, class_folder)
        os.makedirs(output_class_path, exist_ok=True)
        
        if not os.path.exists(class_path):
            print(f"Warning: Folder '{class_path}' not found. Skipping...")
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        if not image_files:
            print(f"Warning: No images found in '{class_path}'. Skipping...")
            continue
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            augment_and_save(image_path, output_class_path, image_file)
    
    print("Image augmentation completed.")

# Example usage
input_folder = "sports_complex"
output_folder = r"sports_complex_augmented"
process_images(input_folder, output_folder)
