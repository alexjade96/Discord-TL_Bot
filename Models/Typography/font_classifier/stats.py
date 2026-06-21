from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix


def plot_curves(results: Dict[str, List], save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(results['train_loss'], label='train')
    ax1.plot(results['val_loss'],   label='val')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax2.plot(results['train_acc'], label='train')
    ax2.plot(results['val_acc'],   label='val')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'[stats] Curves saved to {save_path}')
    plt.show()


def print_report(model, loader, class_names: list, device, top_confused: int = 10):
    model.eval()
    all_logits, all_preds, all_labels = [], [], []
    with torch.inference_mode():
        for X, y in loader:
            logits = model(X.to(device))
            all_logits.append(logits.cpu())
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())

    # Top-K accuracy on the full test set
    logits_cat = torch.cat(all_logits, dim=0)
    labels_t   = torch.tensor(all_labels)
    n_classes  = logits_cat.size(1)
    top1 = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    top3 = logits_cat.topk(min(3, n_classes), dim=1).indices.eq(labels_t.unsqueeze(1)).any(1).float().mean().item()
    top5 = logits_cat.topk(min(5, n_classes), dim=1).indices.eq(labels_t.unsqueeze(1)).any(1).float().mean().item()
    print()
    print(f'--- Test Accuracy  top-1 {top1:.4f}  top-3 {top3:.4f}  top-5 {top5:.4f} ---')

    # Use only the label indices that actually appear (test split may lack some classes)
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
        print(f'  {true_cls:40s} -> {pred_cls:40s}  ({count})')
