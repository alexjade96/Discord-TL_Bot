import json
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    _, topk = logits.topk(min(k, logits.size(1)), dim=1)
    return topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item()


def _gpu_mem_mb(device):
    if device.type != 'cuda':
        return None, None
    alloc    = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return round(alloc, 1), round(reserved, 1)


def _write_heartbeat(path, payload):
    """Best-effort progress snapshot for diagnosing mid-epoch deaths (crash, OOM,
    disconnect) that leave no trace, since last.pt/progress.json only update at
    epoch end. Never allowed to affect training: any failure here is swallowed."""
    if path is None:
        return
    try:
        from datetime import datetime
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps({**payload, 'saved_at': datetime.now().isoformat(timespec='seconds')}, indent=2))
        tmp.replace(path)
    except Exception:
        pass


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam  = np.random.beta(alpha, alpha)
    idx  = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_step(model, loader, loss_fn, optimizer, device,
               mixup_alpha: float = 0.0, clip_grad: float = 1.0,
               heartbeat_path=None, heartbeat_interval: float = 60.0,
               heartbeat_meta: dict = None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_gnorm = 0.0
    n_batches   = len(loader)
    _t0, _t_last_hb = time.monotonic(), 0.0
    for i, (X, y) in enumerate(tqdm(loader, leave=False, desc='  train'), start=1):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if mixup_alpha > 0.0:
            X_mix, y_a, y_b, lam = _mixup_batch(X, y, mixup_alpha)
            logits = model(X_mix)
            loss   = lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
            correct += (logits.argmax(1) == y_a).sum().item()
        else:
            logits = model(X)
            loss   = loss_fn(logits, y)
            correct += (logits.argmax(1) == y).sum().item()

        loss.backward()
        if clip_grad > 0:
            total_gnorm += nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad).item()
        optimizer.step()

        total_loss += loss.item() * len(y)
        total      += len(y)

        _now = time.monotonic()
        if heartbeat_path is not None and (_now - _t_last_hb) >= heartbeat_interval:
            _alloc_mb, _reserved_mb = _gpu_mem_mb(device)
            _write_heartbeat(heartbeat_path, {
                **(heartbeat_meta or {}),
                'phase_name':       'train',
                'batch':            i,
                'total_batches':    n_batches,
                'elapsed_secs':     round(_now - _t0, 1),
                'gpu_mem_alloc_mb': _alloc_mb,
                'gpu_mem_reserved_mb': _reserved_mb,
            })
            _t_last_hb = _now

    return total_loss / total, correct / total, total_gnorm / n_batches if n_batches else 0.0


def eval_step(model, loader, loss_fn, device,
             heartbeat_path=None, heartbeat_interval: float = 60.0,
             heartbeat_meta: dict = None):
    from sklearn.metrics import f1_score, precision_score, recall_score
    model.eval()
    total_loss, correct, top3, top5, total = 0.0, 0, 0, 0, 0
    all_preds, all_labels = [], []
    n_batches = len(loader)
    _t0, _t_last_hb = time.monotonic(), 0.0
    with torch.inference_mode():
        for i, (X, y) in enumerate(tqdm(loader, leave=False, desc='  eval'), start=1):
            X, y   = X.to(device), y.to(device)
            logits = model(X)
            loss   = loss_fn(logits, y)
            preds  = logits.argmax(1)
            total_loss += loss.item() * len(y)
            correct    += (preds == y).sum().item()
            top3       += _topk_correct(logits, y, 3)
            top5       += _topk_correct(logits, y, 5)
            total      += len(y)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

            _now = time.monotonic()
            if heartbeat_path is not None and (_now - _t_last_hb) >= heartbeat_interval:
                _alloc_mb, _reserved_mb = _gpu_mem_mb(device)
                _write_heartbeat(heartbeat_path, {
                    **(heartbeat_meta or {}),
                    'phase_name':       'eval',
                    'batch':            i,
                    'total_batches':    n_batches,
                    'elapsed_secs':     round(_now - _t0, 1),
                    'gpu_mem_alloc_mb': _alloc_mb,
                    'gpu_mem_reserved_mb': _reserved_mb,
                })
                _t_last_hb = _now
    v_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    v_rec  = recall_score(all_labels,   all_preds, average='macro', zero_division=0)
    v_f1   = f1_score(all_labels,       all_preds, average='macro', zero_division=0)
    return total_loss / total, correct / total, top3 / total, top5 / total, v_prec, v_rec, v_f1


def train_loop(
    model, train_loader, val_loader, loss_fn, optimizer,
    epochs: int, device, writer=None, epoch_offset: int = 0,
    mixup_alpha: float = 0.0, clip_grad: float = 1.0,
    scheduler=None, on_epoch_end=None,
    heartbeat_path=None, heartbeat_interval: float = 60.0,
) -> Dict[str, List]:
    results = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'val_top3':   [], 'val_top5':  [],
        'precision':  [], 'recall':    [], 'f1': [],
        'grad_norm':  [], 'lr':        [],
    }
    for epoch in range(1, epochs + 1):
        g = epoch + epoch_offset
        t_loss, t_acc, t_gnorm                         = train_step(model, train_loader, loss_fn, optimizer, device,
                                                                     mixup_alpha=mixup_alpha, clip_grad=clip_grad,
                                                                     heartbeat_path=heartbeat_path, heartbeat_interval=heartbeat_interval,
                                                                     heartbeat_meta={'epoch': g})
        v_loss, v_acc, v_top3, v_top5, v_prec, v_rec, v_f1 = eval_step(model, val_loader, loss_fn, device,
                                                                     heartbeat_path=heartbeat_path, heartbeat_interval=heartbeat_interval,
                                                                     heartbeat_meta={'epoch': g})
        _lr = optimizer.param_groups[-1]['lr']

        results['train_loss'].append(t_loss)
        results['train_acc'].append(t_acc)
        results['val_loss'].append(v_loss)
        results['val_acc'].append(v_acc)
        results['val_top3'].append(v_top3)
        results['val_top5'].append(v_top5)
        results['precision'].append(v_prec)
        results['recall'].append(v_rec)
        results['f1'].append(v_f1)
        results['grad_norm'].append(t_gnorm)
        results['lr'].append(_lr)

        print(
            f'Epoch {g:3d} | '
            f'train loss {t_loss:.4f}  acc {t_acc:.4f}  gnorm {t_gnorm:.3f} | '
            f'val loss {v_loss:.4f}  acc {v_acc:.4f}  prec {v_prec:.4f}  rec {v_rec:.4f}  f1 {v_f1:.4f} | '
            f'lr {_lr:.2e}'
        )
        if writer:
            writer.add_scalars('Loss',     {'train': t_loss, 'val': v_loss}, g)
            writer.add_scalars('Accuracy', {'train': t_acc,  'val': v_acc,
                                            'val_top3': v_top3, 'val_top5': v_top5}, g)
            writer.add_scalars('PRF1',     {'precision': v_prec, 'recall': v_rec, 'f1': v_f1}, g)
            writer.add_scalar('GradNorm',  t_gnorm, g)
            writer.add_scalar('LR',        _lr, g)
        if scheduler:
            scheduler.step()
        if on_epoch_end:
            on_epoch_end(g, v_acc, {
                'train_loss': t_loss, 'train_acc': t_acc,
                'val_loss':   v_loss, 'val_acc':   v_acc,
                'val_top3':   v_top3, 'val_top5':  v_top5,
                'precision':  v_prec, 'recall':    v_rec, 'f1': v_f1,
                'grad_norm':  t_gnorm,
                'lr':         _lr,
            })
    return results
