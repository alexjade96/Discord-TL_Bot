from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix


def plot_curves(results: Dict[str, List], save_path: str = None):
    has_f1 = bool(results.get('f1'))
    has_lr = bool(results.get('lr'))
    n_cols = 2 + int(has_f1) + int(has_lr)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    col = 0

    axes[col].plot(results['train_loss'], label='train')
    axes[col].plot(results['val_loss'],   label='val')
    axes[col].set_title('Loss'); axes[col].set_xlabel('Epoch'); axes[col].legend()
    col += 1

    axes[col].plot(results['train_acc'], label='train acc')
    axes[col].plot(results['val_acc'],   label='val acc')
    axes[col].set_title('Accuracy'); axes[col].set_xlabel('Epoch'); axes[col].legend()
    col += 1

    if has_f1:
        axes[col].plot(results['f1'], label='val f1 (macro)', color='tab:green')
        if results.get('precision'):
            axes[col].plot(results['precision'], label='precision', linestyle='--', alpha=0.7)
        if results.get('recall'):
            axes[col].plot(results['recall'],    label='recall',    linestyle=':', alpha=0.7)
        axes[col].set_title('Val Precision / Recall / F1'); axes[col].set_xlabel('Epoch'); axes[col].legend()
        col += 1

    if has_lr:
        axes[col].plot(results['lr'], color='tab:orange')
        axes[col].set_title('Learning Rate'); axes[col].set_xlabel('Epoch')
        axes[col].set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'[stats] Curves saved to {save_path}')
    plt.close(fig)


def print_report(model, loader, class_names: list, device, top_confused: int = 10):
    model.eval()
    all_logits, all_preds, all_labels = [], [], []
    with torch.inference_mode():
        for batch, y in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            all_logits.append(logits.cpu())
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())

    if not all_labels:
        print('[stats] Test split is empty — nothing to report.')
        return

    logits_cat = torch.cat(all_logits, dim=0)
    labels_t   = torch.tensor(all_labels)
    n_classes  = logits_cat.size(1)
    top1 = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    top3 = logits_cat.topk(min(3, n_classes), dim=1).indices.eq(
        labels_t.unsqueeze(1)).any(1).float().mean().item()
    print()
    print(f'--- Test Accuracy  top-1 {top1:.4f}  top-3 {top3:.4f} ---')
    print(f'--- Chance baseline (uniform over {n_classes} authors): {1 / n_classes:.4f} ---')

    present       = sorted(set(all_labels) | set(all_preds))
    present_names = [class_names[i] for i in present]
    print()
    print('--- Classification Report ---')
    print(classification_report(
        all_labels, all_preds,
        labels=present, target_names=present_names, zero_division=0,
    ))
    cm = confusion_matrix(all_labels, all_preds, labels=present)
    n  = len(present)
    pairs = sorted(
        [(cm[i, j], present_names[i], present_names[j])
         for i in range(n)
         for j in range(n)
         if i != j and cm[i, j] > 0],
        reverse=True,
    )
    print(f'--- Top {top_confused} Confused Pairs (true -> predicted) ---')
    for count, true_cls, pred_cls in pairs[:top_confused]:
        print(f'  {true_cls:30s} -> {pred_cls:30s}  ({count})')
