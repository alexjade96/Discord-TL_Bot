# CLI entry point for character classifier training.
# Run from OCR/:
#   python -m char_classifier.train
#   python -m char_classifier.train --backbone convnext_tiny --epochs 40
#   python -m char_classifier.train --backbone dinov2_vitb14 --epochs 30   # GPU recommended
#   python -m char_classifier.train --resume checkpoints/latin/last.pt --epochs 30
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

if __package__:
    from .data          import get_dataloaders
    from .engine        import train_loop
    from .model_builder import create_model
    from .model_utils   import print_param_summary, unfreeze_backbone
    from .stats         import plot_curves, print_report
    from .utils         import (get_device, load_checkpoint, peek_checkpoint_epoch,
                                save_checkpoint, set_seed)
else:
    from data          import get_dataloaders
    from engine        import train_loop
    from model_builder import create_model
    from model_utils   import print_param_summary, unfreeze_backbone
    from stats         import plot_curves, print_report
    from utils         import (get_device, load_checkpoint, peek_checkpoint_epoch,
                               save_checkpoint, set_seed)


_HERE         = Path(__file__).parent
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
    p.add_argument('--epochs',           type=int,   default=48,
                   help='Total epochs including freeze warm-up (default 48; Latin best at 42)')
    p.add_argument('--freeze-epochs',    type=int,   default=3,
                   help='Head-only warm-up epochs before fine-tuning backbone '
                        '(default 3; phase 1 contributes <16%% val acc regardless of length)')
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
    p.add_argument('--mixup-alpha',      type=float, default=0.2,
                   help='MixUp alpha (Beta distribution param). 0 = disabled '
                        '(default 0.2; 0.4 created a 10-15pt train/val gap with no accuracy gain)')
    p.add_argument('--scheduler',        default='cosine',
                   choices=['cosine', 'cosine-warm', 'none'],
                   help='Phase-2 LR scheduler. cosine=CosineAnnealingLR (original); '
                        'cosine-warm=CosineAnnealingWarmRestarts(T_0=15,T_mult=2) — '
                        'preferred for small datasets (kana/hangul/cjk); none=constant LR')
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
    p.add_argument('--select-metric',    default='val_acc', choices=['val_acc', 'f1'],
                   help='Metric that decides which epoch becomes best.pt (default val_acc). '
                        'Use f1 when classes are imbalanced.')
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--num-workers',      type=int,   default=0)
    p.add_argument('--resume',           default=None,
                   help='Resume from last.pt or best.pt. Pass the same --epochs as the '
                        'original run so the phase boundary and scheduler position are correct.')
    p.add_argument('--no-tensorboard',   action='store_true')
    return p.parse_args()


def _make_p1_optimizer(model, lr):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )


def _make_p2_optimizer(model, lr):
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'head' not in n and 'classifier' not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and ('head' in n or 'classifier' in n)]
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params,     'lr': lr},
    ])


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
    json.dump(
        {'backbone': args.backbone, 'scripts': list(scripts), 'epochs': args.epochs,
         'freeze_epochs': args.freeze_epochs, 'scheduler': args.scheduler,
         'mixup_alpha': args.mixup_alpha},
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

    freeze_epochs = min(args.freeze_epochs, args.epochs)
    fine_epochs   = args.epochs - freeze_epochs

    # --- Determine resume phase before creating any optimizer ---
    start_epoch, best_val_acc = 0, 0.0
    resume_into_p2 = False
    if args.resume:
        resume_epoch   = peek_checkpoint_epoch(args.resume)
        resume_into_p2 = (resume_epoch >= freeze_epochs)
        print(f'[train] Checkpoint epoch={resume_epoch}  '
              f'resuming into {"phase 2" if resume_into_p2 else "phase 1"}')

    # --- Build model; unfreeze backbone now if resuming into phase 2 ---
    model = create_model(args.backbone, num_classes=len(class_names), freeze_base=True).to(device)
    if resume_into_p2:
        unfreeze_backbone(model, args.backbone, n_blocks=args.unfreeze_blocks)
    print_param_summary(model)

    # --- Create the right optimizer so its state can be restored ---
    if resume_into_p2:
        optimizer = _make_p2_optimizer(model, args.lr)
    else:
        optimizer = _make_p1_optimizer(model, args.lr)

    # --- Load checkpoint (restores model weights + optimizer state) ---
    if args.resume:
        start_epoch, best_val_acc, _ = load_checkpoint(args.resume, model, optimizer, device)
        print(f'[train] Resumed from epoch {start_epoch}, best val acc {best_val_acc:.4f}')

    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(ckpt_dir / 'runs'))

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    all_results = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'val_top3':   [], 'val_top5':  [],
        'precision':  [], 'recall':    [], 'f1': [],
        'grad_norm':  [], 'lr':        [],
    }

    # --- Shared epoch-end callback: saves last.pt every epoch, best.pt on improvement ---
    # Checkpoint selection metric. char-dataset is broadly balanced, so val_acc
    # remains the default here; --select-metric f1 is available for skewed sets.
    _sel = args.select_metric
    best_score_ref    = [best_val_acc]
    current_optimizer = [optimizer]

    _progress_path = ckpt_dir / 'progress.json'
    _epoch_log: list = []
    _best_epoch_ref = [start_epoch]
    try:
        _prior_history = json.loads(_progress_path.read_text()).get('history', [])
    except Exception:
        _prior_history = []

    # Protect best.pt's score across runs, even a fresh (non-resumed) one: some
    # scripts (Latin, CJK — see Cell 2) are deliberately restarted from scratch
    # each session, and best.pt must keep the best-ever result across restarts
    # rather than being overwritten by this run's own first, still-untrained
    # epoch beating an initial best_score_ref of 0.
    if _prior_history:
        _prior_best = max(_prior_history, key=lambda e: e.get(_sel, 0.0))
        if _prior_best.get(_sel, 0.0) > best_score_ref[0]:
            best_score_ref  = [_prior_best.get(_sel, best_val_acc)]
            _best_epoch_ref = [_prior_best.get('epoch', start_epoch)]

    # Seed the logged history only when actually resuming — otherwise a fresh
    # run's epoch numbers collide with a prior run's and duplicate progress.json's
    # history (and mislabel epochs_since_best against a run this one isn't
    # continuing).
    if args.resume and _prior_history:
        _epoch_log = _prior_history
        for _e in _epoch_log:
            for k in all_results:
                if k in _e:
                    all_results[k].append(_e[k])

    _t_ref = [time.monotonic()]

    def on_epoch_end(g, v_acc, metrics=None):
        from datetime import datetime
        _now        = time.monotonic()
        _epoch_secs = round(_now - _t_ref[0], 1)
        _t_ref[0]   = _now

        _phase  = 1 if g <= freeze_epochs else 2
        _lr     = round(float(metrics['lr']),        8) if metrics else 0.0
        _gnorm  = round(float(metrics['grad_norm']), 4) if metrics else 0.0
        _prec   = round(float(metrics['precision']), 6) if metrics else 0.0
        _rec    = round(float(metrics['recall']),    6) if metrics else 0.0
        _f1     = round(float(metrics['f1']),        6) if metrics else 0.0
        _overfit = round(float(metrics['train_acc'] - v_acc), 6) if metrics else 0.0
        _prev_val_acc    = _epoch_log[-1]['val_acc'] if _epoch_log else float(v_acc)
        _val_acc_delta   = round(float(v_acc) - _prev_val_acc, 6)
        _epochs_since_best = g - _best_epoch_ref[0]
        _epochs_remaining  = args.epochs - g

        _meta = {
            'epoch':             g,
            'total_epochs':      args.epochs,
            'epochs_remaining':  _epochs_remaining,
            'phase':             _phase,
            'phase_label':       'head warm-up' if _phase == 1 else 'backbone fine-tune',
            'val_acc':           round(float(v_acc), 6),
            'best_score':        round(float(best_score_ref[0]), 6),
            'best_epoch':        _best_epoch_ref[0],
            'epochs_since_best': _epochs_since_best,
            'f1':                _f1,
            'lr':                _lr,
            'epoch_secs':        _epoch_secs,
            'scripts':           list(scripts),
            'backbone':          args.backbone,
            'freeze_epochs':     args.freeze_epochs,
            'saved_at':          datetime.now().isoformat(timespec='seconds'),
        }
        save_checkpoint(model, current_optimizer[0], g, v_acc,
                        ckpt_dir / 'last.pt', class_names, meta=_meta)
        _score = _f1 if _sel == 'f1' else float(v_acc)
        if _score > best_score_ref[0]:
            best_score_ref[0]  = _score
            _best_epoch_ref[0] = g
            _meta['best_score'] = round(float(_score), 6)
            _meta['best_epoch']   = g
            save_checkpoint(model, current_optimizer[0], g, v_acc,
                            ckpt_dir / 'best.pt', class_names, meta=_meta)
            print(f'[train] New best ({_sel}): {_score:.4f}  '
                  f'[acc {v_acc:.4f}  f1 {_f1:.4f}] -> saved best.pt')
        if metrics:
            _epoch_log.append({
                'epoch':             g,
                'lr':                _lr,
                'grad_norm':         _gnorm,
                'epoch_secs':        _epoch_secs,
                'overfit_gap':       _overfit,
                'val_acc_delta':     _val_acc_delta,
                'epochs_since_best': _epochs_since_best,
                **{k: round(float(v), 6)
                   for k, v in metrics.items()
                   if k not in ('lr', 'grad_norm')},
            })
        _progress_path.write_text(json.dumps({
            'scripts':           list(scripts),
            'backbone':          args.backbone,
            'total_epochs':      args.epochs,
            'freeze_epochs':     args.freeze_epochs,
            'completed':         g,
            'epochs_remaining':  _epochs_remaining,
            'phase':             _phase,
            'phase_label':       'head warm-up' if _phase == 1 else 'backbone fine-tune',
            'best_score':        round(float(best_score_ref[0]), 6),
            'best_epoch':        _best_epoch_ref[0],
            'epochs_since_best': _epochs_since_best,
            'last_val_acc':      round(float(v_acc), 6),
            'last_val_acc_delta': _val_acc_delta,
            'last_f1':           _f1,
            'last_precision':    _prec,
            'last_recall':       _rec,
            'last_lr':           _lr,
            'last_grad_norm':    _gnorm,
            'last_epoch_secs':   _epoch_secs,
            'eta_secs':          round(_epoch_secs * _epochs_remaining),
            'saved_at':          datetime.now().isoformat(timespec='seconds'),
            'history':           _epoch_log,
        }, indent=2))

    try:
        # ---- Phase 1: head warm-up (skip if resuming into phase 2) ----
        if not resume_into_p2 and freeze_epochs > 0:
            remaining_p1 = freeze_epochs - start_epoch
            if remaining_p1 > 0:
                print(f'[train] Phase 1 - head warm-up '
                      f'({remaining_p1} of {freeze_epochs} epoch(s) remaining)'
                      f'  mixup={args.mixup_alpha}  clip_grad={args.clip_grad}')
                results = train_loop(
                    model, train_loader, val_loader, loss_fn, optimizer,
                    epochs=remaining_p1, device=device, writer=writer,
                    epoch_offset=start_epoch,
                    mixup_alpha=args.mixup_alpha, clip_grad=args.clip_grad,
                    on_epoch_end=on_epoch_end,
                )
                for k in all_results:
                    all_results[k].extend(results[k])

        # ---- Phase 2: fine-tune with unfrozen backbone blocks ----
        if fine_epochs > 0:
            if not resume_into_p2:
                # Entering phase 2 for the first time — unfreeze and swap optimizer
                unfreeze_backbone(model, args.backbone, n_blocks=args.unfreeze_blocks)
                print_param_summary(model)
                optimizer = _make_p2_optimizer(model, args.lr)
                current_optimizer[0] = optimizer

            if args.scheduler == 'cosine-warm':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=15, T_mult=2
                )
            elif args.scheduler == 'none':
                scheduler = None
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=fine_epochs
                )
            # Fast-forward scheduler to match elapsed phase-2 epochs
            p2_elapsed = max(start_epoch - freeze_epochs, 0) if resume_into_p2 else 0
            if scheduler is not None:
                for _ in range(p2_elapsed):
                    scheduler.step()

            p2_start    = max(start_epoch, freeze_epochs)
            remaining_p2 = args.epochs - p2_start
            if remaining_p2 > 0:
                print(f'[train] Phase 2 - fine-tuning '
                      f'({remaining_p2} of {fine_epochs} epoch(s) remaining, '
                      f'{args.unfreeze_blocks} blocks)'
                      f'  scheduler={args.scheduler}  mixup={args.mixup_alpha}  clip_grad={args.clip_grad}')
                results = train_loop(
                    model, train_loader, val_loader, loss_fn, optimizer,
                    epochs=remaining_p2, device=device, writer=writer,
                    epoch_offset=p2_start,
                    mixup_alpha=args.mixup_alpha, clip_grad=args.clip_grad,
                    scheduler=scheduler, on_epoch_end=on_epoch_end,
                )
                for k in all_results:
                    all_results[k].extend(results[k])

    except KeyboardInterrupt:
        last_pt = ckpt_dir / 'last.pt'
        print(f'\n[train] Interrupted. Best {_sel} so far: {best_score_ref[0]:.4f} '
              f'(epoch {_best_epoch_ref[0]})')
        if last_pt.exists():
            saved_epoch = peek_checkpoint_epoch(str(last_pt))
            print(f'[train] last.pt saved at epoch {saved_epoch}: {last_pt}')
            print(f'[train] Resume with:')
            print(f'  python -m char_classifier.train '
                  f'--scripts {" ".join(scripts)} '
                  f'--epochs {args.epochs} '
                  f'--resume {last_pt}')
        else:
            print('[train] No last.pt — no epoch completed before interrupt.')
        if writer:
            writer.close()
        if all_results['train_loss']:
            plot_curves(all_results, save_path=str(ckpt_dir / 'curves.png'))
        return

    if writer:
        writer.close()

    print(f'[train] Best {_sel}: {best_score_ref[0]:.4f} (epoch {_best_epoch_ref[0]})')

    # Evaluate the checkpoint deploy/compare will actually load, not the live
    # final-epoch model. They differ whenever the best epoch is not the last.
    best_pt = ckpt_dir / 'best.pt'
    if best_pt.exists():
        load_checkpoint(best_pt, model, device=device)
        print(f'[train] Evaluating best.pt (epoch {_best_epoch_ref[0]}) on test set ...')
    else:
        print('[train] No best.pt - evaluating the final-epoch model on test set ...')
    print_report(model, test_loader, class_names, device)
    plot_curves(all_results, save_path=str(ckpt_dir / 'curves.png'))


if __name__ == '__main__':
    main()
