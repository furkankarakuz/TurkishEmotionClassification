import torch
from torch import nn
from typing import Type
from transformers import BertTokenizer


def save_model(model: nn.Module, save_dir: str) -> None:
    """Save a model's weights to disk.

    The state dict is made contiguous before saving and written to ``{save_dir}/pytorch_model.bin``.

    Args:
        model: The trained PyTorch model whose weights should be saved.
        save_dir: Path to the directory where the weights file is written. The directory is expected to already exist.

    Returns:
        None.
    """

    state_dict = model.state_dict()
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    torch.save(state_dict, f"{save_dir}/pytorch_model.bin")

    print(f"Model ve tokenizer '{save_dir}' klasörüne kaydedildi.")


def load_model(model_class: Type[nn.Module], bert_model_name: str, num_classes: int, tokenizer_dir: str) -> tuple[nn.Module, BertTokenizer]:
    """Load a tokenizer and a model with saved weights.

    The tokenizer is loaded from ``tokenizer_dir`` and the model weights are loaded from ``{tokenizer_dir}/pytorch_model.bin``. The model is moved to the best available device (MPS if available, otherwise CPU) and set to eval mode.

    Args:
        model_class: The model class to instantiate (e.g. ``BERTClassifier``).
        bert_model_name: Name or path of the underlying BERT base model (e.g. ``"dbmdz/bert-base-turkish-uncased"``).
        num_classes: Number of output classes the model was trained with.
        tokenizer_dir: Directory containing the saved tokenizer and the ``pytorch_model.bin`` weights file.

    Returns:
        A tuple ``(model, tokenizer)`` with the loaded model (on device and in eval mode) and the loaded tokenizer.
    """

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Kullanılan cihaz: {device}")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

    # Build model
    model = model_class(bert_model_name, num_classes)

    # Load model weights
    model.load_state_dict(torch.load(f"{tokenizer_dir}/pytorch_model.bin", map_location=device))

    # Move the model to the device and switch to eval mode
    model.to(device)
    model.eval()

    print("✅ The model and tokenizer have been loaded.")

    return model, tokenizer
