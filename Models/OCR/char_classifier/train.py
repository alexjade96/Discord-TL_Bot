# CLI entry point for character classifier training.
# Run from OCR/:
#   python -m char_classifier.train
#   python -m char_classifier.train --backbone convnext_tiny --epochs 40
#   python -m char_classifier.train --backbone dinov2_vitb14 --epochs 30   # GPU recommended
#   python -m char_classifier.train --resume ../checkpoints/best.pt
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

if __package__:
    from .data          import get_dataloaders
    from .engine        import train_loop
    from .model_builder import create_model
    from .model_utils   import print_param_summary, unfreeze_backbone
    from .stats         import plot_curves, print_report
    from .utils         import get_device, load_checkpoint, save_checkpoint, set_seed
else:
    from data          import get_dataloaders
    from engine        import train_loop
    from model_builder import create_model
    from model_utils   import print_param_summary, unfreeze_backbone
    from stats         import plot_curves, print_report
    from utils         import get_device, load_checkpoint, save_checkpoint, set_seed


_HERE        = Path(__file__).parent
_DATASET_ROOT = _HERE.parent.parent / 'Datasets' / 'char-dataset'
_DEFAULT_CKPTS = str(_HERE.parent / 'checkpoints')

_SCRIPT_NAMES = ('latin', 'kana', 'hangul', 'cjk')


def parse_args():
    p = argparse.ArgumentParser(description='Character OCR Classifier Training')
    p.add_argument('--scripts', nargs='+', default=['latin'],
                   choices=[*_SCRIPT_NAMES, 'all'],
                   help='Script subdirs to include. "all" expands to latin kana hangul cjk '
                        '(default: latin)')
    p.add_argument('--checkpoint-dir',   default=_DEFAULT_CKPTS)
    p.add_argument('--backbone',         default='dinov2_vits14',
                   choices=['dinov2_vits14', 'dinov2_vitb14', 'convnext_tiny'],
                   help='dinov2_vitb14 is higher-capacity but needs a GPU (86M params)')
    p.add_argument('--epochs',           type=int,   default=30)
    p.add_argument('--freeze-epochs',    type=int,   default=5,
                   help='Head-only warm-up epochs before fine-tuning backbone')
    p.add_argument('--unfreeze-blocks',  type=int,   default=4,
                   help='Backbone blocks/stages to unfreeze in phase 2')
    p.add_argument('--batch-size',       type=int,   default=32)
    p.add_argument('--lr',               type=float, default=1e-3,
                   help='Head LR; backbone uses lr * 0.1 in phase 2')
    p.add_argument('--augment',          default='heavy',
                   choices=['none', 'light', 'heavy'])
    p.add_argument('--grid-mode',        default='single',
                   choices=['single', 'rotated', 'all'],
                   help='single=TileGrid3x3 only | rotated=+full-grid rotation | '
                        'all=random choice among all 6 variants per sample')
    p.add_argument('--mixup-alpha',      type=float, default=0.4,
                   help='MixUp alpha (Beta distribution param). 0 = disabled')
    p.add_argument('--clip-grad',        type=float, default=1.0,
                   help='Max gradient norm for clipping. 0 = disabled')
    p.add_argument('--max-per-class',   type=int,   default=0,
                   help='Cap each class at N images before splitting (0 = no cap). '
                        'Use for fast smoke tests, e.g. --max-per-class 20.')
    p.add_argument('--min-per-class',   type=int,   default=5,
                   help='Minimum images to include a class (default 5). '
                        'Lower to 1 after re-rendering CJK with more sizes.')
    p.add_argument('--no-weighted-sampler', action='store_true',
                   help='Disable class-balanced WeightedRandomSampler (use shuffle instead)')
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--num-workers',      type=int,   default=0)
    p.add_argument('--resume',           default=None)
    p.add_argument('--no-tensorboard',   action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f'[train] Device: {device}')

    scripts = _SCRIPT_NAMES if 'all' in args.scripts else args.scripts

    # Auto-scope checkpoint dir: single script → checkpoints/<script>/
    default_ckpts = Path(_DEFAULT_CKPTS)
    if args.checkpoint_dir == _DEFAULT_CKPTS and len(scripts) == 1:
        ckpt_dir = default_ckpts / scripts[0]
    else:
        ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = [str(_DATASET_ROOT / s) for s in scripts]
    print(f'[train] Scripts: {", ".join(scripts)}')

    # Write config so compare.py knows which backbone to load for this script set
    import json as _json
    _json.dump(
        {'backbone': args.backbone, 'scripts': list(scripts), 'epochs': args.epochs},
        open(ckpt_dir / 'config.json', 'w'), indent=2,
    )

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        dataset_dirs=dataset_dirs,
        batch_size=args.batch_size,
        augment=args.augment,
        grid_mode=args.grid_mode,
        num_workers=args.num_workers,
        seed=args.seed,
        weighted_sampler=not args.no_weighted_sampler,
        max_per_class=args.max_per_class,
        min_per_class=args.min_per_class,
    )
    json.dump(class_names, open(ckpt_dir / 'class_names.json', 'w'), indent=2)

    model = create_model(args.backbone, num_classes=len(class_names), freeze_base=True).to(device)
    print_param_summary(model)

    start_epoch, best_val_acc = 0, 0.0
    all_results = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'val_top3':   [], 'val_top5':  [],
    }

    if args.resume:
        start_epoch, best_val_acc, _ = load_checkpoint(args.resume, model, device=device)
        print(f'[train] Resumed from epoch {start_epoch}, best val acc {best_val_acc:.4f}')

    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(ckpt_dir / 'runs'))

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: head warm-up with frozen backbone
    freeze_epochs = min(args.freeze_epochs, args.epochs)
    if freeze_epochs > 0:
        print(f'[train] Phase 1 - head warm-up ({freeze_epochs} epochs)'
              f'  mixup={args.mixup_alpha}  clip_grad={args.clip_grad}')
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
        results = train_loop(
            model, train_loader, val_loader, loss_fn, optimizer,
            epochs=freeze_epochs, device=device, writer=writer, epoch_offset=start_epoch,
            mixup_alpha=args.mixup_alpha, clip_grad=args.clip_grad,
        )
        for k in all_results:
            all_results[k].extend(results[k])
        best_phase1 = max(results['val_acc'])
        if best_phase1 > best_val_acc:
            best_val_acc = best_phase1
            save_checkpoint(model, optimizer, freeze_epochs + start_epoch,
                            best_val_acc, ckpt_dir / 'best.pt', class_names)

    # Phase 2: fine-tune with unfrozen backbone blocks
    fine_epochs = args.epochs - freeze_epochs
    if fine_epochs > 0:
        print(f'[train] Phase 2 - fine-tuning ({fine_epochs} epochs, {args.unfreeze_blocks} blocks)'
              f'  mixup={args.mixup_alpha}  clip_grad={args.clip_grad}')
        unfreeze_backbone(model, args.backbone, n_blocks=args.unfreeze_blocks)
        print_param_summary(model)

        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and 'head' not in n and 'classifier' not in n]
        head_params     = [p for n, p in model.named_parameters()
                           if p.requires_grad and ('head' in n or 'classifier' in n)]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': head_params,     'lr': args.lr},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_epochs)

        results = train_loop(
            model, train_loader, val_loader, loss_fn, optimizer,
            epochs=fine_epochs, device=device, writer=writer,
            epoch_offset=start_epoch + freeze_epochs,
            mixup_alpha=args.mixup_alpha, clip_grad=args.clip_grad,
        )
        for k in all_results:
            all_results[k].extend(results[k])
        for i, val_acc in enumerate(results['val_acc']):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, optimizer, start_epoch + freeze_epochs + i + 1,
                                best_val_acc, ckpt_dir / 'best.pt', class_names)
            scheduler.step()

    if writer:
        writer.close()

    print(f'[train] Best val acc: {best_val_acc:.4f}')
    print('[train] Evaluating on test set ...')
    print_report(model, test_loader, class_names, device)
    plot_curves(all_results, save_path=str(ckpt_dir / 'curves.png'))


if __name__ == '__main__':
    main()
