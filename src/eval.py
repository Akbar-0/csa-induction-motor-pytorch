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

    # Dataset with the same settings used in training (channels, aux, windowing)
    label_map = {c: i for i, c in enumerate(classes)}
    ds = MotorCurrentDataset(
        data_path,
        window_length=cfg.get('window_length', 3000),
        hop_length=cfg.get('hop_length', 1500),
        label_map=label_map,
        prefer_voltage=cfg.get('prefer_voltage', False),
        include_aux=cfg.get('include_aux', False),
        scaling_type=cfg.get('scaling_type', 'zscore'),
        return_severity=cfg.get('severity_head', False),
    )

    # Infer input channels from a sample to match training
    sample_x, *_ = ds[0]
    in_channels = int(sample_x.shape[0])

    model = create_model({
        'in_channels': in_channels,
        'num_classes': len(classes),
        'dropout': cfg.get('dropout', 0.5),
        'base_channels': cfg.get('base_channels', 64),
        'severity_head': cfg.get('severity_head', False),
        'severity_classes': cfg.get('severity_classes', 7)
    })
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.get('batch_size', 32), collate_fn=collate_fn)

    y_true = []
    y_pred = []
    severity_hist = None
    with torch.no_grad():
        sev_preds = []
        for batch in loader:
            if cfg.get('severity_head', False):
                x, y, sev = batch
            else:
                x, y = batch
            x = x.to(device)
            outputs = model(x)
            if cfg.get('severity_head', False):
                logits, sev_logits = outputs
                sev_idx = sev_logits.argmax(dim=1).cpu().numpy().tolist()
                sev_preds.extend([s + 1 for s in sev_idx])  # map to 1..6
            else:
                logits = outputs
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

    metrics = {'accuracy': float(acc), 'precision_per_class': prec.tolist(), 'recall_per_class': rec.tolist(), 'f1_per_class': f1.tolist(), 'classes': classes, 'severity_head': bool(cfg.get('severity_head', False))}
    if cfg.get('severity_head', False):
        # Save a simple severity histogram plot
        plt.figure(figsize=(6,4))
        sns.histplot(sev_preds, bins=np.arange(1,8)-0.5, kde=False)
        plt.xticks([1,2,3,4,5,6])
        plt.xlabel('Predicted Severity (1-6)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'severity_hist.png'))
        plt.close()
    return metrics