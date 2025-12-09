"""Training orchestration for MCSA."""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data import create_datasets, collate_fn
from src.model import create_model
from src.utils import save_checkpoint, set_seed, save_json


class Augmentor:
    """
    Top-level callable augmentor that's safe to pickle across processes.
    Instantiate with a plain dict of config options, e.g.:
      Augmentor({'noise': True, 'noise_snr_db': 30, 'scale': True})
    """
    def __init__(self, cfg: Dict | None):
        # ensure cfg is a plain dict (picklable)
        self.cfg = dict(cfg) if cfg is not None else {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        L = x.shape[1]

        if cfg.get('noise', False):
            snr_db = cfg.get('noise_snr_db', 30)
            sig_pow = np.mean(x ** 2)
            noise_pow = sig_pow / (10 ** (snr_db / 10))
            x = x + np.random.randn(*x.shape) * np.sqrt(noise_pow)

        if cfg.get('scale', False):
            scale = np.random.uniform(0.9, 1.1, size=(x.shape[0], 1))
            x = x * scale

        if cfg.get('time_shift', False):
            # random integer in [-0.05*L, 0.05*L)
            shift = np.random.randint(-int(0.05 * L), int(0.05 * L))
            x = np.roll(x, shift, axis=1)

        if cfg.get('freq_aug', False):
            Xf = np.fft.rfft(x, axis=1)
            mag = np.abs(Xf)
            phase = np.angle(Xf)
            mag = mag * (1 + 0.02 * np.random.randn(*mag.shape))
            Xf_new = mag * np.exp(1j * phase)
            x = np.fft.irfft(Xf_new, n=L, axis=1)

        return x


class Trainer:
    def __init__(self, cfg: Dict, data_path: str):
        self.cfg = cfg
        set_seed(cfg.get('seed', 42))
        self.save_dir = cfg.get('save_dir', 'runs/exp')
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('use_gpu', True) else 'cpu')
        classes = cfg.get('classes', ["Healthy", "Bearing", "BrokenRotorBar", "StatorShort"])
        label_map = {c: i for i, c in enumerate(classes)}
        self.label_map = label_map
        self.classes = classes

        augment_fn = None
        if cfg.get("augment", {}).get("enabled", False):
            augment_cfg = cfg["augment"]
            augment_fn = Augmentor(augment_cfg)

        ds_train, ds_val, ds_test = create_datasets(
            data_path,
            ratios=tuple(cfg.get('split_ratios', [0.7, 0.15, 0.15])),
            seed=cfg.get('seed', 42),
            window_length=cfg.get('window_length', 3000),
            hop_length=cfg.get('hop_length', 1500),
            label_map=label_map,
            prefer_voltage=cfg.get('prefer_voltage', False),
            file_level_label=cfg.get('file_level_label', False),
            include_aux=cfg.get('include_aux', False),
            scaler=None,
            scaling_type=cfg.get('scaling_type', 'zscore'),
            augment_fn=augment_fn,
            return_severity=cfg.get('severity_head', False),
        )
        self.ds_train, self.ds_val, self.ds_test = ds_train, ds_val, ds_test

        self.train_sampler = None
        if cfg.get('oversample', False):
            labels = ds_train.window_labels
            counts = np.bincount(labels, minlength=len(classes))
            weights = [1.0 / counts[l] if counts[l] > 0 else 0.0 for l in labels]
            self.train_sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

        batch_size = cfg.get('batch_size', 32)
        num_workers = cfg.get('num_workers', 0)
        self.loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=(self.train_sampler is None), sampler=self.train_sampler, num_workers=num_workers, collate_fn=collate_fn)
        self.loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        self.loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

        # Infer input channels from dataset sample (accounts for aux channels)
        sample = ds_train[0]
        x_sample = sample[0] if isinstance(sample, tuple) else sample
        in_ch = int(x_sample.shape[0])
        self.model = create_model({
            'in_channels': in_ch,
            'num_classes': len(classes),
            'dropout': cfg.get('dropout', 0.5),
            'base_channels': cfg.get('base_channels', 64),
            'severity_head': cfg.get('severity_head', False),
            'severity_classes': cfg.get('severity_classes', 6)
        })
        self.model.to(self.device)

        class_weights = None
        if cfg.get('class_weights', None):
            class_weights = torch.tensor(cfg['class_weights'], dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Severity head: we will support classification (CrossEntropy) and regression (SmoothL1)
        self.use_severity = cfg.get('severity_head', False)
        self.severity_loss_cls = nn.CrossEntropyLoss(ignore_index=-100)  # for classification-style severity outputs
        self.severity_loss_reg = nn.SmoothL1Loss(reduction='mean')      # for regression-style severity outputs

        lr_value = cfg.get('lr', 1e-3)
        try:
            lr_value = float(lr_value)
        except Exception:
            lr_value = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_value)
        sched_type = cfg.get('scheduler', 'plateau')
        try:
            epochs_for_sched = int(cfg.get('epochs', 30))
        except Exception:
            epochs_for_sched = 30
        if sched_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs_for_sched)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=cfg.get('plateau_patience', 5))

        # AMP: allocate scaler only if needed, and use the recommended API
        self.use_amp = cfg.get('mixed_precision', False) and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(device_type="cuda", enabled=True)
        else:
            self.scaler = None

        self.tb_writer = None
        if cfg.get('tensorboard', False):
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'tb'))

        self.patience = cfg.get('early_stop_patience', 10)
        self.best_metric = -float('inf')
        self.best_epoch = 0

        resume_path = cfg.get('resume_path')
        if resume_path and os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if 'scaler' in ckpt and self.use_amp and self.scaler is not None:
                self.scaler.load_state_dict(ckpt['scaler'])
            print(f"Resumed from {resume_path} (epoch {ckpt.get('epoch')})")

    def fit(self):
        epochs = self.cfg.get('epochs', 30)
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(1, epochs + 1):
            self.model.train()
            losses = []
            ys_true = []
            ys_pred = []
            pbar = tqdm(self.loader_train, desc=f"Train {epoch}/{epochs}")
            for batch in pbar:
                if not self.use_severity:
                    x, y = batch
                    sev = None
                else:
                    x, y, sev = batch

                x = x.to(self.device)
                y = y.to(self.device)
                if sev is not None:
                    sev = sev.to(self.device)

                # autocast with device type
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(x)
                    if not self.use_severity:
                        logits = outputs
                        loss = self.criterion(logits, y)
                    else:
                        # Expect outputs either (logits, sev_logits) or (logits) if model not returning severity
                        if isinstance(outputs, tuple) and len(outputs) == 2:
                            logits, sev_logits = outputs
                        else:
                            # model did not return severity; treat as regular classification
                            logits = outputs
                            sev_logits = None

                        loss_cls = self.criterion(logits, y)

                        # default severity loss term
                        loss_sev = 0.0
                        alpha = float(self.cfg.get('severity_loss_weight', 0.3))

                        if sev_logits is not None:
                            # Determine whether sev_logits are classification logits (C>1) or regression (C==1 or 1-dim)
                            if sev_logits.dim() == 1 or (sev_logits.dim() == 2 and sev_logits.size(1) == 1):
                                # regression output (B,) or (B,1)
                                # ensure sev_targets are floats normalized to [0,1]
                                if sev.dtype in (torch.long, torch.int):
                                    # if ints likely 0..6 or 1..6: convert to floats / normalize
                                    sev_float = sev.to(torch.float32)
                                    # Heuristic: if max > 1, assume 1..6 and map dividing by 6
                                    if float(sev_float.max().item()) > 1.0:
                                        sev_float = sev_float / 6.0
                                else:
                                    sev_float = sev.to(torch.float32)
                                # mask out healthy samples
                                mask = (y != self.label_map['Healthy'])
                                if mask.any():
                                    # squeeze pred if needed
                                    
                                    # pred = sev_logits.squeeze()
                                    # pred_masked = pred[mask]
                                    # target_masked = sev_float[mask].to(pred_masked.dtype)
                                    # loss_sev = self.severity_loss_reg(pred_masked, target_masked)
                                    # reshape to (B,) without dropping batch dimension
                                    
                                    pred = sev_logits.reshape(-1)

                                    # apply mask
                                    pred_masked = pred[mask]
                                    target_masked = sev_float.reshape(-1)[mask]

                                    if pred_masked.numel() > 0:
                                        loss_sev = self.severity_loss_reg(pred_masked, target_masked)
                                    else:
                                        loss_sev = torch.tensor(0.0, device=self.device)
                                else:
                                    loss_sev = 0.0
                            else:
                                # classification-style severity head (B, C)
                                num_sev_classes = sev_logits.size(1)
                                # Build class targets (integers 0..num_sev_classes-1), with ignore_index for healthy
                                if sev.dtype in (torch.float32, torch.float16, torch.float64):
                                    # assume normalized floats in [0,1]. Map to classes by rounding to (num_classes-1)
                                    sev_float = sev.to(torch.float32)
                                    # if values appear >1.0 (not normalized), try to rescale by dividing by 6
                                    if float(sev_float.max().item()) > 1.0:
                                        sev_float = sev_float / 6.0
                                    sev_idx = torch.clamp(torch.round(sev_float * (num_sev_classes - 1)).long(), 0, num_sev_classes - 1)
                                else:
                                    # integer targets: could be 0..6 or 1..6
                                    sev_idx = sev.clone().long()
                                    # if values contain >=(num_sev_classes), assume 1..6 mapping -> subtract 1
                                    if int(sev_idx.max().item()) >= num_sev_classes:
                                        # Convert 1..6 -> 0..5
                                        sev_idx = torch.where(sev_idx > 0, sev_idx - 1, sev_idx)
                                # ignore healthy (set to -100)
                                sev_targets = torch.full_like(sev_idx, -100)
                                mask_fault = (y != self.label_map['Healthy'])
                                if mask_fault.any():
                                    sev_targets[mask_fault] = sev_idx[mask_fault]
                                loss_sev = self.severity_loss_cls(sev_logits, sev_targets)
                        else:
                            # no sev_logits returned by model; severity loss = 0
                            loss_sev = 0.0

                        # total loss
                        if isinstance(loss_sev, torch.Tensor):
                            loss = loss_cls + alpha * loss_sev
                        else:
                            loss = loss_cls

                # backward + step (handle AMP or not)
                self.optimizer.zero_grad()
                if self.use_amp and self.scaler is not None:
                    # scale and step
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.cfg.get('grad_clip', 5.0)))
                    self.optimizer.step()

                losses.append(loss.item())
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                ys_pred.extend(preds.tolist())
                ys_true.extend(y.detach().cpu().numpy().tolist())
                pbar.set_postfix(loss=np.mean(losses), acc=accuracy_score(ys_true, ys_pred))

            train_acc = accuracy_score(ys_true, ys_pred)
            train_loss = float(np.mean(losses))
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            val_metrics = self.evaluate_loader(self.loader_val)
            val_acc = val_metrics['accuracy']
            val_loss = val_metrics['loss']
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            if self.tb_writer:
                self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
                self.tb_writer.add_scalar('Loss/val', val_loss, epoch)
                self.tb_writer.add_scalar('Acc/train', train_acc, epoch)
                self.tb_writer.add_scalar('Acc/val', val_acc, epoch)

            ckpt_state = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'cfg': self.cfg,
                'scaler': self.scaler.state_dict() if (self.use_amp and self.scaler is not None) else None,
                'label_map': self.label_map,
            }
            save_checkpoint(ckpt_state, os.path.join(self.save_dir, f'epoch{epoch}.pth'))
            improved = val_acc > self.best_metric
            if improved:
                self.best_metric = val_acc
                self.best_epoch = epoch
                save_checkpoint(ckpt_state, os.path.join(self.save_dir, 'best.pth'))
            if epoch - self.best_epoch >= self.patience:
                print(f"Early stopping at epoch {epoch} (best {self.best_metric:.4f} at {self.best_epoch})")
                break

        eval_batch = self.cfg.get('eval_batch_size', self.cfg.get('batch_size', 32))
        train_eval_loader = DataLoader(self.ds_train, batch_size=eval_batch, shuffle=False, num_workers=self.cfg.get('num_workers', 0), collate_fn=collate_fn)
        train_metrics = self.evaluate_loader(train_eval_loader, return_confusion=True, split_name='train')
        val_metrics = self.evaluate_loader(self.loader_val, return_confusion=True, split_name='val')
        test_metrics = self.evaluate_loader(self.loader_test, return_confusion=True, split_name='test')

        summary = {
            'best_val_acc': self.best_metric,
            'best_epoch': self.best_epoch,
            'test_acc': test_metrics['accuracy'],
            'history': history,
            'confusion_matrices': {
                'train': train_metrics.get('confusion_matrix'),
                'val': val_metrics.get('confusion_matrix'),
                'test': test_metrics.get('confusion_matrix'),
            },
            'confusion_paths': {
                'train': train_metrics.get('confusion_path'),
                'val': val_metrics.get('confusion_path'),
                'test': test_metrics.get('confusion_path'),
            },
        }
        save_json(os.path.join(self.save_dir, 'summary.json'), summary)
        self._plot_curves(history)

    def _plot_curves(self, history: Dict):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(self.save_dir,'loss_curves.png')); plt.close()
        plt.figure(figsize=(8,4))
        plt.plot(history['train_acc'], label='train_acc')
        plt.plot(history['val_acc'], label='val_acc')
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(self.save_dir,'acc_curves.png')); plt.close()

    def evaluate_loader(self, loader: DataLoader, return_confusion: bool = False, split_name: str | None = None) -> Dict:
        self.model.eval()
        ys_true = []
        ys_pred = []
        losses = []
        with torch.no_grad():
            for batch in loader:
                if not self.use_severity:
                    x, y = batch
                    sev = None
                else:
                    x, y, sev = batch

                x = x.to(self.device)
                y = y.to(self.device)
                if sev is not None:
                    sev = sev.to(self.device)

                outputs = self.model(x)
                if not self.use_severity:
                    logits = outputs
                    loss = self.criterion(logits, y)
                else:
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        logits, sev_logits = outputs
                    else:
                        logits = outputs
                        sev_logits = None

                    loss_cls = self.criterion(logits, y)

                    loss_sev = 0.0
                    alpha = float(self.cfg.get('severity_loss_weight', 0.3))

                    if sev_logits is not None:
                        if sev_logits.dim() == 1 or (sev_logits.dim() == 2 and sev_logits.size(1) == 1):
                            # regression
                            if sev.dtype in (torch.long, torch.int):
                                sev_float = sev.to(torch.float32)
                                if float(sev_float.max().item()) > 1.0:
                                    sev_float = sev_float / 6.0
                            else:
                                sev_float = sev.to(torch.float32)
                            mask = (y != self.label_map['Healthy'])
                            if mask.any():
                                
                                # pred = sev_logits.squeeze()
                                # pred_masked = pred[mask]
                                # target_masked = sev_float[mask].to(pred_masked.dtype)
                                # loss_sev = self.severity_loss_reg(pred_masked, target_masked)
                                # reshape to (B,) without dropping batch dimension
                                pred = sev_logits.reshape(-1)
                                
                                # apply mask
                                pred_masked = pred[mask]
                                target_masked = sev_float.reshape(-1)[mask]
                                
                                if pred_masked.numel() > 0:
                                    loss_sev = self.severity_loss_reg(pred_masked, target_masked)
                                else:
                                    loss_sev = torch.tensor(0.0, device=self.device)
                            else:
                                loss_sev = 0.0
                        else:
                            # classification
                            num_sev_classes = sev_logits.size(1)
                            if sev.dtype in (torch.float32, torch.float16, torch.float64):
                                sev_float = sev.to(torch.float32)
                                if float(sev_float.max().item()) > 1.0:
                                    sev_float = sev_float / 6.0
                                sev_idx = torch.clamp(torch.round(sev_float * (num_sev_classes - 1)).long(), 0, num_sev_classes - 1)
                            else:
                                sev_idx = sev.clone().long()
                                if int(sev_idx.max().item()) >= num_sev_classes:
                                    sev_idx = torch.where(sev_idx > 0, sev_idx - 1, sev_idx)
                            sev_targets = torch.full_like(sev_idx, -100)
                            mask_fault = (y != self.label_map['Healthy'])
                            if mask_fault.any():
                                sev_targets[mask_fault] = sev_idx[mask_fault]
                            loss_sev = self.severity_loss_cls(sev_logits, sev_targets)
                    else:
                        loss_sev = 0.0

                    if isinstance(loss_sev, torch.Tensor):
                        loss = loss_cls + alpha * loss_sev
                    else:
                        loss = loss_cls

                losses.append(loss.item())
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                ys_pred.extend(preds.tolist())
                ys_true.extend(y.detach().cpu().numpy().tolist())

        acc = accuracy_score(ys_true, ys_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(ys_true, ys_pred, labels=list(range(len(self.label_map))), zero_division=0)

        result = {
            'accuracy': acc,
            'loss': float(np.mean(losses)),
            'precision': prec.tolist(),
            'recall': rec.tolist(),
            'f1': f1.tolist(),
        }

        if return_confusion:
            cm = confusion_matrix(ys_true, ys_pred, labels=list(range(len(self.label_map))))
            result['confusion_matrix'] = cm.tolist()
            if split_name:
                result['confusion_path'] = self._save_confusion_plot(cm, split_name)

        return result

    def _save_confusion_plot(self, cm: np.ndarray, split_name: str) -> str:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.classes, yticklabels=self.classes, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {split_name}')
        plt.tight_layout()
        out_path = os.path.join(self.save_dir, f'confusion_{split_name}.png')
        plt.savefig(out_path)
        plt.close()
        return out_path
