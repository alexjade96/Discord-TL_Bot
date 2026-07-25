# CLI entry point for authorship classifier training.
# Run from Models/UserRecognition/:
#   python -m author_classifier.train --guild 1502045408677986405
#   python -m author_classifier.train --guild <ID> --backbone distilbert-base-multilingual-cased
#   python -m author_classifier.train --guild <ID> --resume checkpoints/<ID>/last.pt --epochs 20
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

if __package__:
    from .data          import get_dataloaders
    from .engine        import train_loop
    from .model_builder import BACKBONE_CHOICES, create_model, create_tokenizer
    from .model_utils   import print_param_summary, unfreeze_encoder
    from .stats         import plot_curves, print_report
    from .utils         import (get_device, load_checkpoint, peek_checkpoint_epoch,
                                save_checkpoint, set_seed)
else:
    from data          import get_dataloaders
    from engine        import train_loop
    from model_builder import BACKBONE_CHOICES, create_model, create_tokenizer
    from model_utils   import print_param_summary, unfreeze_encoder
    from stats         import plot_curves, print_report
    from utils         import (get_device, load_checkpoint, peek_checkpoint_epoch,
                               save_checkpoint, set_seed)


_HERE          = Path(__file__).parent                                    # author_classifier/
_DATASET_ROOT  = _HERE.parent.parent / 'Datasets' / 'chat-dataset'
_DEFAULT_CKPTS = str(_HERE.parent / 'checkpoints')


def parse_args():
    p = argparse.ArgumentParser(description='Authorship Attribution Classifier Training')
    p.add_argument('--guild', required=True,
                   help='Guild ID — selects chat-dataset/<guild_id>/ and checkpoints/<guild_id>/')
    p.add_argument('--dataset-dir',    default=None,
                   help='Override the chat-dataset guild directory')
    p.add_argument('--checkpoint-dir', default=None,
                   help='Override checkpoints/<guild_id>/')
    p.add_argument('--backbone',       default='xlm-roberta-base',
                   choices=list(BACKBONE_CHOICES),
                   help='xlm-roberta-base is multilingual and the default; '
                        'distilbert-base-multilingual-cased is ~2x faster on CPU')
    p.add_argument('--epochs',         type=int,   default=20,
                   help='Total epochs including freeze warm-up (default 20)')
    p.add_argument('--freeze-epochs',  type=int,   default=3,
                   help='Head-only warm-up epochs before unfreezing the encoder (default 3)')
    p.add_argument('--unfreeze-layers', type=int,  default=4,
                   help='Encoder layers to unfreeze in phase 2 (default 4)')
    p.add_argument('--batch-size',     type=int,   default=16)
    p.add_argument('--max-length',     type=int,   default=256,
                   help='Token cap per sample (default 256)')
    p.add_argument('--lr',             type=float, default=1e-3,
                   help='Head LR; encoder uses lr * 0.05 in phase 2')
    p.add_argument('--augment',        default='light',
                   choices=['none', 'light', 'heavy'],
                   help='Token-level augmentation. Style-preserving only.')
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--scheduler',      default='cosine',
                   choices=['cosine', 'cosine-warm', 'none'])
    p.add_argument('--clip-grad',      type=float, default=1.0,
                   help='Max gradient norm. 0 = disabled')
    p.add_argument('--no-weighted-sampler', action='store_true',
                   help='Disable class-balanced sampling (use shuffle instead)')
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--num-workers',    type=int,   default=0)
    p.add_argument('--resume',         default=None,
                   help='Resume from last.pt or best.pt. Pass the same --epochs as the '
                        'original run so the phase boundary and scheduler position line up.')
    p.add_argument('--no-tensorboard', action='store_true')
    return p.parse_args()


def _make_p1_optimizer(model, lr):
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def _make_p2_optimizer(model, lr):
    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith('encoder')]
    head_params    = [p for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith('head')]
    # 0.05 rather than the image pipeline's 0.1: a pretrained language encoder
    # fine-tuned on a few hundred short samples drifts fast.
    return torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr * 0.05},
        {'params': head_params,    'lr': lr},
    ])


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f'[train] Device: {device}')

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else _DATASET_ROOT / str(args.guild)
    if not dataset_dir.exists():
        raise SystemExit(
            f'[train] No dataset at {dataset_dir}\n'
            f'        Build it first:  python Models/Datasets/build_chat.py --guild {args.guild}'
        )

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir \
        else Path(_DEFAULT_CKPTS) / str(args.guild)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f'[train] Dataset:     {dataset_dir}')
    print(f'[train] Checkpoints: {ckpt_dir}')

    # Surface the dataset build's warnings here too — they describe limits that
    # no amount of training will fix.
    meta_path = dataset_dir / 'meta.json'
    if meta_path.exists():
        ds_meta = json.loads(meta_path.read_text(encoding='utf-8'))
        for w in ds_meta.get('warnings', []):
            print(f'[train] DATASET WARNING: {w}')

    tokenizer = create_tokenizer(args.backbone)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        dataset_dir=dataset_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        augment=args.augment,
        num_workers=args.num_workers,
        weighted_sampler=not args.no_weighted_sampler,
    )

    # Written at training start so deploy.py and predict.py can rebuild the
    # right architecture without guessing.
    json.dump(
        {'backbone': args.backbone, 'guild_id': str(args.guild),
         'num_classes': len(class_names), 'max_length': args.max_length,
         'epochs': args.epochs, 'freeze_epochs': args.freeze_epochs,
         'scheduler': args.scheduler},
        open(ckpt_dir / 'config.json', 'w'), indent=2,
    )
    json.dump(class_names, open(ckpt_dir / 'class_names.json', 'w'), indent=2)

    freeze_epochs = min(args.freeze_epochs, args.epochs)
    fine_epochs   = args.epochs - freeze_epochs

    start_epoch, best_val_acc = 0, 0.0
    resume_into_p2 = False
    if args.resume:
        resume_epoch   = peek_checkpoint_epoch(args.resume)
        resume_into_p2 = (resume_epoch >= freeze_epochs)
        print(f'[train] Checkpoint epoch={resume_epoch}  '
              f'resuming into {"phase 2" if resume_into_p2 else "phase 1"}')

    model = create_model(args.backbone, num_classes=len(class_names), freeze_base=True).to(device)
    if resume_into_p2:
        unfreeze_encoder(model, args.backbone, n_layers=args.unfreeze_layers)
    print_param_summary(model)

    optimizer = (_make_p2_optimizer if resume_into_p2 else _make_p1_optimizer)(model, args.lr)

    if args.resume:
        start_epoch, best_val_acc, _ = load_checkpoint(args.resume, model, optimizer, device)
        print(f'[train] Resumed from epoch {start_epoch}, best val acc {best_val_acc:.4f}')

    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(ckpt_dir / 'runs'))

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    all_results = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [], 'val_top3': [],
        'precision':  [], 'recall':    [], 'f1': [],
        'grad_norm':  [], 'lr':        [],
    }

    best_val_acc_ref  = [best_val_acc]
    current_optimizer = [optimizer]

    _progress_path = ckpt_dir / 'progress.json'
    try:
        _prior = json.loads(_progress_path.read_text())
        _epoch_log: list = _prior.get('history', [])
        for _e in _epoch_log:
            for k in all_results:
                if k in _e:
                    all_results[k].append(_e[k])
        _best_epoch_ref = [max(_epoch_log, key=lambda e: e.get('val_acc', 0.0)).get('epoch', start_epoch)
                           if _epoch_log else start_epoch]
    except Exception:
        _epoch_log = []
        _best_epoch_ref = [start_epoch]

    _t_ref = [time.monotonic()]

    def on_epoch_end(g, v_acc, metrics=None):
        from datetime import datetime
        _now        = time.monotonic()
        _epoch_secs = round(_now - _t_ref[0], 1)
        _t_ref[0]   = _now

        _phase   = 1 if g <= freeze_epochs else 2
        _lr      = round(float(metrics['lr']),        8) if metrics else 0.0
        _gnorm   = round(float(metrics['grad_norm']), 4) if metrics else 0.0
        _f1      = round(float(metrics['f1']),        6) if metrics else 0.0
        _overfit = round(float(metrics['train_acc'] - v_acc), 6) if metrics else 0.0
        _prev    = _epoch_log[-1]['val_acc'] if _epoch_log else float(v_acc)
        _delta   = round(float(v_acc) - _prev, 6)
        _epochs_remaining = args.epochs - g

        _meta = {
            'epoch':            g,
            'total_epochs':     args.epochs,
            'epochs_remaining': _epochs_remaining,
            'phase':            _phase,
            'phase_label':      'head warm-up' if _phase == 1 else 'encoder fine-tune',
            'val_acc':          round(float(v_acc), 6),
            'best_val_acc':     round(float(best_val_acc_ref[0]), 6),
            'best_epoch':       _best_epoch_ref[0],
            'f1':               _f1,
            'lr':               _lr,
            'epoch_secs':       _epoch_secs,
            'guild_id':         str(args.guild),
            'backbone':         args.backbone,
            'max_length':       args.max_length,
            'freeze_epochs':    args.freeze_epochs,
            'saved_at':         datetime.now().isoformat(timespec='seconds'),
        }
        save_checkpoint(model, current_optimizer[0], g, v_acc,
                        ckpt_dir / 'last.pt', class_names, meta=_meta)
        if v_acc > best_val_acc_ref[0]:
            best_val_acc_ref[0] = v_acc
            _best_epoch_ref[0]  = g
            _meta['best_val_acc'] = round(float(v_acc), 6)
            _meta['best_epoch']   = g
            save_checkpoint(model, current_optimizer[0], g, v_acc,
                            ckpt_dir / 'best.pt', class_names, meta=_meta)
            print(f'[train] New best: {v_acc:.4f}  f1 {_f1:.4f} -> saved best.pt')
        if metrics:
            _epoch_log.append({
                'epoch':         g,
                'lr':            _lr,
                'grad_norm':     _gnorm,
                'epoch_secs':    _epoch_secs,
                'overfit_gap':   _overfit,
                'val_acc_delta': _delta,
                **{k: round(float(v), 6) for k, v in metrics.items()
                   if k not in ('lr', 'grad_norm')},
            })
        _progress_path.write_text(json.dumps({
            'guild_id':         str(args.guild),
            'backbone':         args.backbone,
            'total_epochs':     args.epochs,
            'freeze_epochs':    args.freeze_epochs,
            'completed':        g,
            'epochs_remaining': _epochs_remaining,
            'phase':            _phase,
            'phase_label':      'head warm-up' if _phase == 1 else 'encoder fine-tune',
            'best_val_acc':     round(float(best_val_acc_ref[0]), 6),
            'best_epoch':       _best_epoch_ref[0],
            'last_val_acc':     round(float(v_acc), 6),
            'last_f1':          _f1,
            'last_lr':          _lr,
            'last_epoch_secs':  _epoch_secs,
            'eta_secs':         round(_epoch_secs * _epochs_remaining),
            'saved_at':         datetime.now().isoformat(timespec='seconds'),
            'history':          _epoch_log,
        }, indent=2))

    try:
        # ---- Phase 1: head warm-up ----
        if not resume_into_p2 and freeze_epochs > 0:
            remaining_p1 = freeze_epochs - start_epoch
            if remaining_p1 > 0:
                print(f'[train] Phase 1 - head warm-up '
                      f'({remaining_p1} of {freeze_epochs} epoch(s) remaining)')
                results = train_loop(
                    model, train_loader, val_loader, loss_fn, optimizer,
                    epochs=remaining_p1, device=device, writer=writer,
                    epoch_offset=start_epoch, clip_grad=args.clip_grad,
                    on_epoch_end=on_epoch_end,
                )
                for k in all_results:
                    all_results[k].extend(results[k])

        # ---- Phase 2: fine-tune with unfrozen encoder layers ----
        if fine_epochs > 0:
            if not resume_into_p2:
                unfreeze_encoder(model, args.backbone, n_layers=args.unfreeze_layers)
                print_param_summary(model)
                optimizer = _make_p2_optimizer(model, args.lr)
                current_optimizer[0] = optimizer

            if args.scheduler == 'cosine-warm':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=max(5, fine_epochs // 3), T_mult=2)
            elif args.scheduler == 'none':
                scheduler = None
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=fine_epochs)

            p2_elapsed = max(start_epoch - freeze_epochs, 0) if resume_into_p2 else 0
            if scheduler is not None:
                for _ in range(p2_elapsed):
                    scheduler.step()

            p2_start     = max(start_epoch, freeze_epochs)
            remaining_p2 = args.epochs - p2_start
            if remaining_p2 > 0:
                print(f'[train] Phase 2 - fine-tuning '
                      f'({remaining_p2} of {fine_epochs} epoch(s) remaining, '
                      f'{args.unfreeze_layers} layers)  scheduler={args.scheduler}')
                results = train_loop(
                    model, train_loader, val_loader, loss_fn, optimizer,
                    epochs=remaining_p2, device=device, writer=writer,
                    epoch_offset=p2_start, clip_grad=args.clip_grad,
                    scheduler=scheduler, on_epoch_end=on_epoch_end,
                )
                for k in all_results:
                    all_results[k].extend(results[k])

    except KeyboardInterrupt:
        last_pt = ckpt_dir / 'last.pt'
        print(f'\n[train] Interrupted. Best val acc so far: {best_val_acc_ref[0]:.4f}')
        if last_pt.exists():
            print(f'[train] last.pt saved at epoch {peek_checkpoint_epoch(str(last_pt))}: {last_pt}')
            print(f'[train] Resume with:')
            print(f'  python -m author_classifier.train --guild {args.guild} '
                  f'--epochs {args.epochs} --resume {last_pt}')
        else:
            print('[train] No last.pt - no epoch completed before interrupt.')
        if writer:
            writer.close()
        if all_results['train_loss']:
            plot_curves(all_results, save_path=str(ckpt_dir / 'curves.png'))
        return

    if writer:
        writer.close()

    print(f'[train] Best val acc: {best_val_acc_ref[0]:.4f}')
    print('[train] Evaluating on test set ...')
    print_report(model, test_loader, class_names, device)
    plot_curves(all_results, save_path=str(ckpt_dir / 'curves.png'))


if __name__ == '__main__':
    main()
