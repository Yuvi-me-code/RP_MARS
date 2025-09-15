import cv2
import matplotlib.pyplot as plt

def visualize_canny(image_path, sigma=1.0, low_threshold=100, high_threshold=200):
    # Read image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print(f"Error: Unable to read image {image_path}")
        return

    # Apply Canny edge detection
    edges = cv2.Canny(image_gray, low_threshold, high_threshold)

    # Plot original grayscale and edge-detected side by side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edges")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = r"C:/Users\Yuvraj\OneDrive\Files\Work\MARS\RP\3.JPG"  # replace with your photo path
    visualize_canny(image_path)
