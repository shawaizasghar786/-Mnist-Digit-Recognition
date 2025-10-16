from load_data import load_images, load_labels
from process import preprocess_data
from train_model import train_classifier
from evaluate_model import evaluate


# Load data
train_images = load_images("MNIST/train-images.idx3-ubyte")
train_labels = load_labels("MNIST/train-labels.idx1-ubyte")
test_images = load_images("MNIST/t10k-images.idx3-ubyte")
test_labels = load_labels("MNIST/t10k-labels.idx1-ubyte")


# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(train_images, test_images, train_labels, test_labels)

# Train
model = train_classifier(X_train, y_train)

# Evaluate
evaluate(model, X_test, y_test)
