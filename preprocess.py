import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.GaussianBlur(img, (5,5), 0)  # Noise removal
    img = cv2.resize(img, (128, 128))  # Resize for model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Reshape for CNN
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img