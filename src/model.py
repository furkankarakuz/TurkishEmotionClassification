from typing import Optional
from torch import nn, Tensor
from transformers import BertModel


class BERTClassifier(nn.Module):
    """A BERT-based text classifier.

    Wraps a pretrained BERT encoder with a dropout layer and a linear classification head that maps the pooled ``[CLS]`` representation to class logits.
    """

    def __init__(self, bert_model_name: str, num_classes: int) -> None:
        """Initialize the classifier.

        Args:
            bert_model_name: Name or path of the pretrained BERT model to load (e.g. ``"dbmdz/bert-base-turkish-uncased"``).
            num_classes: Number of output classes.
        """
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Optional[Tensor] = None) -> Tensor | tuple[Tensor, Tensor]:
        """Run a forward pass.

        Args:
            input_ids: Token id tensor of shape ``(batch_size, seq_len)``.
            attention_mask: Attention mask tensor of shape ``(batch_size, seq_len)``.
            labels: Optional ground-truth label tensor of shape ``(batch_size,)``. When provided, the cross-entropy loss is also computed and returned.

        Returns:
            If ``labels`` is given, a tuple ``(loss, logits)``; otherwise the ``logits`` tensor of shape ``(batch_size, num_classes)``.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits
