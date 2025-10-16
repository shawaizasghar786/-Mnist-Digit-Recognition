def preprocess_data(train_images, test_images, train_labels, test_labels):
    X_train = train_images.reshape(-1, 28*28) / 255.0
    X_test = test_images.reshape(-1, 28*28) / 255.0
    return X_train, X_test, train_labels, test_labels
