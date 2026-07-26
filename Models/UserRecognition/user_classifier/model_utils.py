import torch.nn as nn


def count_params(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_param_summary(model: nn.Module):
    total, trainable = count_params(model)
    print(f'  Total params:     {total:>12,}')
    print(f'  Trainable params: {trainable:>12,}')
    print(f'  Frozen params:    {total - trainable:>12,}')


def _encoder_layers(model: nn.Module, backbone: str):
    """Return the encoder's transformer layer list for the given backbone."""
    enc = model.encoder
    if backbone == 'xlm-roberta-base':
        return list(enc.encoder.layer)
    if backbone == 'distilbert-base-multilingual-cased':
        return list(enc.transformer.layer)
    raise ValueError(f'Unknown backbone {backbone!r}')


def unfreeze_encoder(model: nn.Module, backbone: str, n_layers: int = 4):
    """Unfreeze the top n_layers of the encoder for fine-tuning phase 2."""
    for layer in _encoder_layers(model, backbone)[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    # The pooled representation is a mean over token states, so the final
    # LayerNorm scales it — leave it trainable alongside the top layers.
    for name, module in model.encoder.named_modules():
        if isinstance(module, nn.LayerNorm) and 'layer.' not in name:
            for p in module.parameters():
                p.requires_grad = True

    total, trainable = count_params(model)
    print(f'[unfreeze] {trainable:,} / {total:,} params now trainable')
