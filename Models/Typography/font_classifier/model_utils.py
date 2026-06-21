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


# Unfreeze the last n_blocks of the backbone for fine-tuning phase 2.
# backbone: 'dinov2_vits14' | 'dinov2_vitb14' | 'convnext_tiny'
def unfreeze_backbone(model: nn.Module, backbone: str, n_blocks: int = 4):
    if backbone in ('dinov2_vits14', 'dinov2_vitb14'):
        for block in list(model.backbone.blocks)[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in model.backbone.norm.parameters():
            p.requires_grad = True
    elif backbone == 'convnext_tiny':
        for stage in list(model.features.children())[-n_blocks:]:
            for p in stage.parameters():
                p.requires_grad = True
    _, trainable = count_params(model)
    total = sum(p.numel() for p in model.parameters())
    print(f'[unfreeze] {trainable:,} / {total:,} params now trainable')
