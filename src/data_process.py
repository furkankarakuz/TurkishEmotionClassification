import pandas as pd
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase


def tokenize(batch: dict[str, list[str]], tokenizer: PreTrainedTokenizerBase, max_length: int) -> BatchEncoding:
    """Tokenize a batch of texts without padding.

    Padding is intentionally left out so it can be applied dynamically per batch by a data collator (e.g. ``DataCollatorWithPadding``), which is more efficient than padding every sample to ``max_length``.

    Args:
        batch: A batch mapping produced by ``datasets.Dataset.map`` with ``batched=True``; must contain a ``"text"`` key holding the list of raw text samples.
        tokenizer: HuggingFace tokenizer used to encode the texts.
        max_length: Maximum sequence length; longer sequences are truncated.

    Returns:
        A ``BatchEncoding`` with the tokenized fields (``input_ids``, ``attention_mask`` and, depending on the tokenizer, ``token_type_ids``).
    """
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def make_dataset(dframe: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dataset:
    """Build a tokenized HuggingFace ``Dataset`` from a pandas DataFrame.

    The DataFrame's ``"text"`` column is tokenized and then dropped, and the ``"label"`` column is renamed to ``"labels"`` so the result is ready to be fed directly into a ``Trainer``. No torch formatting is applied; tensor conversion is delegated to the data collator at batch time.

    Args:
        dframe: Source DataFrame containing at least the columns ``"text"`` (raw text) and ``"label"`` (integer-encoded class label).
        tokenizer: HuggingFace tokenizer used to encode the texts.
        max_length: Maximum sequence length passed through to ``tokenize``.

    Returns:
        A ``datasets.Dataset`` with the columns ``input_ids``, ``attention_mask`` and ``labels``.
    """
    ds = Dataset.from_pandas(dframe[["text", "label"]], preserve_index=False)
    ds = ds.map(tokenize, batched=True, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},)
    ds = ds.rename_column("label", "labels")
    return ds
