from sklearn.metrics import accuracy_score, classification_report
import os
os.makedirs("output", exist_ok=True)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print("Classification Report:\n", report)

    with open("output/evaluation.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
