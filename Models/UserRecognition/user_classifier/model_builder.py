"""Transformer encoder + classification head for user recognition.

Mirrors char_classifier/model_builder.py: a factory over several backbones,
each returning a frozen-by-default encoder wrapped in a two-layer MLP head.
"""

import torch
import torch.nn as nn

# backbone name -> (HF model id, hidden size)
_BACKBONES = {
    'xlm-roberta-base':                   ('xlm-roberta-base',                   768),
    'distilbert-base-multilingual-cased': ('distilbert-base-multilingual-cased', 768),
}

BACKBONE_CHOICES = tuple(_BACKBONES)


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token states over the attention mask.

    Preferred over the CLS token here: CLS is only meaningful after it has been
    fine-tuned into a sentence representation, and phase 1 trains with the
    encoder frozen.
    """
    mask = mask.unsqueeze(-1).type_as(hidden)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class TransformerClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int,
                 hidden_size: int = 768, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask, **_ignored):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _masked_mean(out.last_hidden_state, attention_mask)
        return self.head(pooled)


def create_model(backbone: str, num_classes: int, freeze_base: bool = True) -> nn.Module:
    """
    backbone:
      'xlm-roberta-base'                   (278M params, multilingual — default)
      'distilbert-base-multilingual-cased' (134M params, ~2x faster on CPU)
    """
    if backbone not in _BACKBONES:
        raise ValueError(
            f'Unknown backbone {backbone!r}. Choose: {" | ".join(BACKBONE_CHOICES)}.'
        )
    from transformers import AutoModel

    model_id, hidden = _BACKBONES[backbone]
    print(f'[model] Loading {model_id} ...')
    encoder = AutoModel.from_pretrained(model_id)
    if freeze_base:
        for p in encoder.parameters():
            p.requires_grad = False
    return TransformerClassifier(encoder, num_classes, hidden_size=hidden)


def create_tokenizer(backbone: str):
    if backbone not in _BACKBONES:
        raise ValueError(
            f'Unknown backbone {backbone!r}. Choose: {" | ".join(BACKBONE_CHOICES)}.'
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_BACKBONES[backbone][0])
