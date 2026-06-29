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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        total_loss += loss.item() * len(y)
        total      += len(y)

    return total_loss / total, correct / total


def eval_step(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, top3, top5, total = 0.0, 0, 0, 0, 0
    with torch.inference_mode():
        for X, y in tqdm(loader, leave=False, desc='  eval'):
            X, y = X.to(device), y.to(device)
            logits     = model(X)
            loss       = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            top3       += _topk_correct(logits, y, 3)
            top5       += _topk_correct(logits, y, 5)
            total      += len(y)
    return total_loss / total, correct / total, top3 / total, top5 / total


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
    }
    for epoch in range(1, epochs + 1):
        g = epoch + epoch_offset
        t_loss, t_acc                 = train_step(model, train_loader, loss_fn, optimizer, device,
                                                   mixup_alpha=mixup_alpha, clip_grad=clip_grad)
        v_loss, v_acc, v_top3, v_top5 = eval_step(model, val_loader, loss_fn, device)

        results['train_loss'].append(t_loss)
        results['train_acc'].append(t_acc)
        results['val_loss'].append(v_loss)
        results['val_acc'].append(v_acc)
        results['val_top3'].append(v_top3)
        results['val_top5'].append(v_top5)

        print(
            f'Epoch {g:3d} | '
            f'train loss {t_loss:.4f}  acc {t_acc:.4f} | '
            f'val loss {v_loss:.4f}  acc {v_acc:.4f}  top3 {v_top3:.4f}  top5 {v_top5:.4f}'
        )
        if writer:
            writer.add_scalars('Loss',     {'train': t_loss, 'val': v_loss}, g)
            writer.add_scalars('Accuracy', {'train': t_acc,  'val': v_acc,
                                            'val_top3': v_top3, 'val_top5': v_top5}, g)
        if scheduler:
            scheduler.step()
        if on_epoch_end:
            on_epoch_end(g, v_acc)
    return results
