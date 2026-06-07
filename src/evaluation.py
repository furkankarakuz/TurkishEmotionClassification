import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Union
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute evaluation metrics from model predictions.

    Reports accuracy along with macro-averaged precision, recall and F1. Macro averaging weights every class equally regardless of its frequency, which makes it more informative than accuracy on imbalanced datasets.

    Args:
        eval_pred: A tuple ``(logits, labels)`` where ``logits`` has shape ``(num_samples, num_classes)`` and ``labels`` has shape ``(num_samples,)``.

    Returns:
        A dictionary with the keys ``"accuracy"``, ``"f1_macro"``, ``"precision_macro"`` and ``"recall_macro"`` mapping to their float values.
    """
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "f1_macro": f1, "precision_macro": p, "recall_macro": r}


def get_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a dataloader and collect predictions.

    Args:
        model: The model to evaluate. It is switched to eval mode internally.
        dataloader: DataLoader yielding batches with the keys ``input_ids``, ``attention_mask`` and ``labels``.
        device: Device on which to run the forward passes.

    Returns:
        A tuple ``(true_labels, predictions, probs)`` of numpy arrays, where ``true_labels`` and ``predictions`` have shape ``(num_samples,)`` and ``probs`` has shape ``(num_samples, num_classes)``.
    """
    model.eval()
    predictions = []
    true_labels = []
    probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract logits depending on the output type
            if isinstance(outputs, SequenceClassifierOutput):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                logits = outputs

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            true = labels.cpu().numpy()

            predictions.extend(preds)
            probs.extend(prob)
            true_labels.extend(true)

    return np.array(true_labels), np.array(predictions), np.array(probs)


def classification_report_fn(y_true: np.ndarray, y_pred: np.ndarray, label_encoder: LabelEncoder) -> str:
    """Build a text classification report with original string labels.

    Args:
        y_true: Ground-truth labels as encoded integers.
        y_pred: Predicted labels as encoded integers.
        label_encoder: Fitted ``LabelEncoder`` used to map the encoded integers back to their original string labels.

    Returns:
        The formatted classification report as a string.
    """
    # Convert encoded labels back to the original string labels
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    return classification_report(y_true_labels, y_pred_labels)


def predict(text: str, model: nn.Module, tokenizer, device: torch.device, max_length: int = 128, id2label: Optional[dict[int, str]] = None, return_proba: bool = False) -> Union[str, int, tuple[Union[str, int], float]]:
    """Predict the class of a single text sample.

    Args:
        text: Raw input text to classify.
        model: The trained model. It is switched to eval mode internally.
        tokenizer: HuggingFace tokenizer used to encode the text.
        device: Device on which to run the forward pass.
        max_length: Maximum sequence length used for tokenization.
        id2label: Optional mapping from class id to a human-readable label. When provided, the returned label is a string; otherwise it is the integer class id.
        return_proba: If ``True``, also return the confidence (probability) of the predicted class.

    Returns:
        The predicted label (string if ``id2label`` is given, else int). If ``return_proba`` is ``True``, a tuple ``(label, confidence)`` is returned instead.
    """
    model.eval()

    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract logits depending on the output type
        if isinstance(outputs, SequenceClassifierOutput):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[1] if len(outputs) > 1 else outputs[0]
        else:
            logits = outputs

        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    pred_label = id2label[pred_id] if id2label else pred_id

    if return_proba:
        confidence = probs[0][pred_id].item()
        return pred_label, confidence

    return pred_label
