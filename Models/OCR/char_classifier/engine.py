from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    _, topk = logits.topk(min(k, logits.size(1)), dim=1)
    return topk.eq(targets.unsqueeze(1)).any(dim=1).sum().item()


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    lam  = np.random.beta(alpha, alpha)
    idx  = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_step(model, loader, loss_fn, optimizer, device,
               mixup_alpha: float = 0.0, clip_grad: float = 1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_gnorm = 0.0
    for X, y in tqdm(loader, leave=False, desc='  train'):
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

    n_batches = len(loader)
    return total_loss / total, correct / total, total_gnorm / n_batches if n_batches else 0.0


def eval_step(model, loader, loss_fn, device):
    from sklearn.metrics import f1_score, precision_score, recall_score
    model.eval()
    total_loss, correct, top3, top5, total = 0.0, 0, 0, 0, 0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for X, y in tqdm(loader, leave=False, desc='  eval'):
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
    v_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    v_rec  = recall_score(all_labels,   all_preds, average='macro', zero_division=0)
    v_f1   = f1_score(all_labels,       all_preds, average='macro', zero_division=0)
    return total_loss / total, correct / total, top3 / total, top5 / total, v_prec, v_rec, v_f1


def train_loop(
    model, train_loader, val_loader, loss_fn, optimizer,
    epochs: int, device, writer=None, epoch_offset: int = 0,
    mixup_alpha: float = 0.0, clip_grad: float = 1.0,
    scheduler=None, on_epoch_end=None,
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
                                                                     mixup_alpha=mixup_alpha, clip_grad=clip_grad)
        v_loss, v_acc, v_top3, v_top5, v_prec, v_rec, v_f1 = eval_step(model, val_loader, loss_fn, device)
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
