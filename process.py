import numpy as np

def preprocess_data(train_images, test_images, train_labels, test_labels):
    X_train = train_images.reshape(-1, 28*28) / 255.0
    y_train = np.array(train_labels)

    if isinstance(test_images, np.ndarray) and test_images.size > 0:
        X_test = test_images.reshape(-1, 28*28) / 255.0
        y_test = np.array(test_labels)
    else:
        X_test, y_test = None, None

    return X_train, X_test, y_train, y_test
