import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_encoder: LabelEncoder) -> None:
    """Plot a confusion matrix as an annotated heatmap.

    Args:
        y_true: Ground-truth labels as encoded integers.
        y_pred: Predicted labels as encoded integers.
        label_encoder: Fitted ``LabelEncoder`` used to recover the original string labels and the class ordering.

    Returns:
        None. The figure is displayed via ``matplotlib``.
    """
    # Convert encoded y values back to the original string labels
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_auc(y_true: np.ndarray, y_probs: np.ndarray, label_encoder: LabelEncoder) -> None:
    """Plot one-vs-rest ROC curves with AUC scores for each class.

    Args:
        y_true: Ground-truth labels as encoded integers of shape ``(num_samples,)``.
        y_probs: Predicted class probabilities of shape ``(num_samples, num_classes)``.
        label_encoder: Fitted ``LabelEncoder`` providing the class names.

    Returns:
        None. The figure is displayed via ``matplotlib``.
    """
    classes = label_encoder.classes_
    num_classes = len(classes)

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(7, 6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
