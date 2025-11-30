"""Training orchestration for MCSA."""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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


# NOTE: previous version returned a local function which caused pickling errors on Windows.
# Using a top-level callable class (Augmentor) ensures picklability across processes.


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

        augment_fn = None
        if cfg.get("augment", {}).get("enabled", False):
            augment_cfg = cfg["augment"]
            # instantiate top-level Augmentor (picklable)
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

        self.model = create_model({'in_channels': 3, 'num_classes': len(classes), 'dropout': cfg.get('dropout', 0.5), 'base_channels': cfg.get('base_channels', 64)})
        self.model.to(self.device)

        class_weights = None
        if cfg.get('class_weights', None):
            class_weights = torch.tensor(cfg['class_weights'], dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
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

        self.use_amp = cfg.get('mixed_precision', False) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

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
            if 'scaler' in ckpt and self.use_amp:
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
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
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
                'scaler': self.scaler.state_dict() if self.use_amp else None,
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

        test_metrics = self.evaluate_loader(self.loader_test)
        summary = {
            'best_val_acc': self.best_metric,
            'best_epoch': self.best_epoch,
            'test_acc': test_metrics['accuracy'],
            'history': history,
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

    def evaluate_loader(self, loader: DataLoader) -> Dict:
        self.model.eval()
        ys_true = []
        ys_pred = []
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                losses.append(loss.item())
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                ys_pred.extend(preds.tolist())
                ys_true.extend(y.detach().cpu().numpy().tolist())
        acc = accuracy_score(ys_true, ys_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(ys_true, ys_pred, labels=list(range(len(self.label_map))), zero_division=0)
        return {
            'accuracy': acc,
            'loss': float(np.mean(losses)),
            'precision': prec.tolist(),
            'recall': rec.tolist(),
            'f1': f1.tolist(),
        }
