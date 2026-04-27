import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Generate and save confusion matrix.
    """
    os.makedirs("outputs/confusion_matrices", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("outputs/confusion_matrices/confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved successfully.")


def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluate model performance on test dataset.
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds))

    plot_confusion_matrix(
        all_labels,
        all_preds,
        classes=[str(i) for i in range(10)]
    )

    return accuracy