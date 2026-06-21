# Single-image font family inference.
# Run from Typography/font_classifier/:
#   python predict.py --image path/to/image.png
#   python predict.py --image path/to/image.png --model ../checkpoints/best.pt --top-k 10
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

if __package__:
    from .model_builder import create_model
    from .utils         import get_device, load_checkpoint
else:
    from model_builder import create_model
    from utils         import get_device, load_checkpoint

_HERE = Path(__file__).parent
_DEFAULT_MODEL = str(_HERE.parent / 'checkpoints' / 'best.pt')

_EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(
    image_path: str,
    model_path: str = _DEFAULT_MODEL,
    backbone: str   = 'dinov2_vits14',
    top_k: int      = 5,
    device          = None,
) -> list:
    # Return top-k [(font_family, confidence), ...] for an image
    device = device or get_device()
    names_file = Path(model_path).parent / 'class_names.json'
    if not names_file.exists():
        raise FileNotFoundError(f'class_names.json not found at {names_file}')
    class_names = json.load(open(names_file))

    model = create_model(backbone, num_classes=len(class_names), freeze_base=False).to(device)
    load_checkpoint(model_path, model, device=device)
    model.eval()

    img    = Image.open(image_path).convert('RGB')
    tensor = _EVAL_TF(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]

    k = min(top_k, len(class_names))
    values, indices = torch.topk(probs, k)
    return [(class_names[i], round(v.item(), 4)) for i, v in zip(indices, values)]


def main():
    p = argparse.ArgumentParser(description='Font Classifier Inference')
    p.add_argument('--image',    required=True)
    p.add_argument('--model',    default=_DEFAULT_MODEL)
    p.add_argument('--backbone', default='dinov2_vits14',
                   choices=['dinov2_vits14', 'convnext_tiny'])
    p.add_argument('--top-k',   type=int, default=5)
    args = p.parse_args()

    predictions = predict(args.image, args.model, args.backbone, args.top_k)
    print(f'Font predictions for: {args.image}')
    print('-' * 50)
    for rank, (name, conf) in enumerate(predictions, 1):
        bar = chr(9608) * int(conf * 30)
        print(f'  {rank:2d}. {name:40s}  {conf:.2%}  {bar}')


if __name__ == '__main__':
    main()
