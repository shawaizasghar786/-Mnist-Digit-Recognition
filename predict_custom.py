import cv2
import numpy as np
from train_model import train_classifier
from load_data import load_images, load_labels
from process import preprocess_data

# Load and train model
train_images = load_images("MNIST/train-images.idx3-ubyte")
train_labels = load_labels("MNIST/train-labels.idx1-ubyte")
_, _, X_train, y_train = preprocess_data(train_images, [], train_labels, [])
model = train_classifier(X_train, y_train)

# Load custom image
img = cv2.imread("custom_digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.reshape(1, -1)  # Flatten to match MNIST format

# Predict
prediction = model.predict(img)
print("Predicted digit:", prediction[0])
