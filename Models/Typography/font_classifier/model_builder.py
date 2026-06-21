import torch.nn as nn


class DINOv2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int,
                 embed_dim: int = 384, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        # Two-layer MLP head: Linear → BN → GELU → Dropout → Linear
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def create_dinov2(num_classes: int, freeze_base: bool = True) -> DINOv2Classifier:
    print('[model] Loading DINOv2 ViT-S/14 ...')
    backbone = _load_dinov2('dinov2_vits14', freeze_base)
    return DINOv2Classifier(backbone, num_classes, embed_dim=384)


def create_dinov2b(num_classes: int, freeze_base: bool = True) -> DINOv2Classifier:
    print('[model] Loading DINOv2 ViT-B/14 (86M params) ...')
    backbone = _load_dinov2('dinov2_vitb14', freeze_base)
    return DINOv2Classifier(backbone, num_classes, embed_dim=768)


def _load_dinov2(variant: str, freeze_base: bool):
    import torch
    backbone = torch.hub.load('facebookresearch/dinov2', variant, pretrained=True)
    if freeze_base:
        for p in backbone.parameters():
            p.requires_grad = False
    return backbone


def create_convnext_tiny(num_classes: int, freeze_base: bool = True):
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    print('[model] Loading ConvNeXt-Tiny ...')
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    if freeze_base:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# Factory — backbone choices:
#   'dinov2_vits14'  (default, 22M params, 384-dim)
#   'dinov2_vitb14'  (optional, 86M params, 768-dim — better ceiling, needs GPU)
#   'convnext_tiny'  (torchvision, 28M params)
def create_model(backbone: str, num_classes: int, freeze_base: bool = True) -> nn.Module:
    if backbone == 'dinov2_vits14':
        return create_dinov2(num_classes, freeze_base)
    if backbone == 'dinov2_vitb14':
        return create_dinov2b(num_classes, freeze_base)
    if backbone == 'convnext_tiny':
        return create_convnext_tiny(num_classes, freeze_base)
    raise ValueError(f'Unknown backbone {backbone!r}. '
                     f'Choose dinov2_vits14 | dinov2_vitb14 | convnext_tiny.')
