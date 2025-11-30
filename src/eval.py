"""Evaluation utilities: metrics, confusion matrix plotting"""
from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.model import create_model
from src.data import MotorCurrentDataset, collate_fn


def evaluate_checkpoint(checkpoint_path: str, data_path: str, classes: List[str], device: str = None, save_dir: str = 'runs/eval'):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get('cfg', {})
    model = create_model({'in_channels': 3, 'num_classes': len(classes)})
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    ds = MotorCurrentDataset(data_path, window_length=cfg.get('window_length', 3000), hop_length=cfg.get('hop_length', 1500), label_map={c: i for i, c in enumerate(classes)})
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.get('batch_size', 32), collate_fn=collate_fn)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    acc = np.mean(np.array(y_true) == np.array(y_pred)).item()

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    metrics = {'accuracy': float(acc), 'precision_per_class': prec.tolist(), 'recall_per_class': rec.tolist(), 'f1_per_class': f1.tolist(), 'classes': classes}
    return metrics