import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from train_model import train_classifier
from load_data import load_images, load_labels
from process import preprocess_data
import os

# Load and train model
train_images = load_images("MNIST/train-images.idx3-ubyte")
train_labels = load_labels("MNIST/train-labels.idx1-ubyte")
X_train, _, y_train, _ = preprocess_data(train_images, np.array([]), train_labels, [])
model = train_classifier(X_train, y_train)

# Open file browser
Tk().withdraw()  # Hide root window
image_path = filedialog.askopenfilename(title="Select a digit image", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")])

if not image_path:
    print("❌ No image selected. Please try again.")
    exit()

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("❌ Failed to load image. Make sure it's a valid PNG or JPG.")
    exit()

img = cv2.resize(img, (28, 28))
img_flat = img.flatten().reshape(1, -1)

# Predict
prediction = model.predict(img_flat)
print("✅ Predicted digit:", prediction[0])

# Visualize
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {prediction[0]}")
plt.axis('off')
plt.show()
