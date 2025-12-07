"""Dataset and dataloader utilities for MCSA."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple, Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _detect_channels(df: pd.DataFrame, prefer_voltage: bool = False) -> Tuple[List[str], bool]:
    i_candidates = [c for c in ["Ia", "Ib", "Ic", "I1", "I2", "I3"] if c in df.columns]
    v_candidates = [c for c in ["Va", "Vb", "Vc", "V1", "V2", "V3"] if c in df.columns]
    if prefer_voltage and len(v_candidates) >= 3:
        return v_candidates[:3], True
    if len(i_candidates) >= 3:
        return i_candidates[:3], False
    if len(v_candidates) >= 3:
        return v_candidates[:3], True
    raise ValueError("Need at least 3 current or voltage columns (Ia..Ic or Va..Vc)")


def sliding_windows(x: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
    C, T = x.shape
    if T < window_length:
        pad = window_length - T
        x = np.pad(x, ((0, 0), (0, pad)), mode="constant")
        T = window_length
    num = 1 + (T - window_length) // hop_length
    windows = np.stack([x[:, i * hop_length: i * hop_length + window_length] for i in range(num)], axis=0)
    return windows


class MotorCurrentDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str] | str,
        window_length: int = 3000,
        hop_length: int = 1500,
        label_map: Optional[Dict[str, int]] = None,
        prefer_voltage: bool = False,
        file_level_label: bool = False,
        scaler: Optional[Dict] = None,
        scaling_type: str = 'zscore',
        augment_fn=None,
        include_aux: bool = False,
        infer_from_filename: bool = True,
        return_severity: bool = False,
    ):
        if isinstance(paths, str):
            if os.path.isdir(paths):
                self.files = [os.path.join(paths, f) for f in sorted(os.listdir(paths)) if f.endswith('.csv')]
            elif os.path.isfile(paths):
                self.files = [paths]
            else:
                raise ValueError(f"Path not found: {paths}")
        else:
            self.files = list(paths)

        self.window_length = window_length
        self.hop_length = hop_length
        self.label_map = label_map or {"Healthy": 0, "Bearing": 1, "BrokenRotorBar": 2, "StatorShort": 3}
        self.prefer_voltage = prefer_voltage
        self.file_level_label = file_level_label
        self.scaler = scaler
        self.scaling_type = scaling_type
        self.augment_fn = augment_fn
        self.include_aux = include_aux
        self.infer_from_filename = infer_from_filename
        self.return_severity = return_severity

        self.index = []
        self.metadata = []
        self.window_labels = []

        for fi, f in enumerate(self.files):
            df = pd.read_csv(f)
            ch_names, is_voltage = _detect_channels(df, prefer_voltage=self.prefer_voltage)
            core_data = df[ch_names].to_numpy().T
            aux_data = []
            aux_cols = []
            if self.include_aux:
                for col in ["rpm", "RPM", "vibration", "Torque"]:
                    if col in df.columns:
                        aux_cols.append(col)
                        aux_data.append(df[col].to_numpy())
            if aux_data:
                aux_arr = np.stack(aux_data, axis=0)
                data = np.concatenate([core_data, aux_arr], axis=0)
            else:
                data = core_data

            windows = sliding_windows(data, self.window_length, self.hop_length)
            n_win = windows.shape[0]

            if self.file_level_label and 'label' in df.columns:
                file_label_str = df['label'].dropna().astype(str).iloc[0]
                label_int = self.label_map.get(file_label_str, 0)
                win_labels = [label_int] * n_win
            elif 'label' in df.columns:
                sample_labels = df['label'].astype(str).to_numpy()
                win_labels = []
                for wi in range(n_win):
                    start = wi * self.hop_length
                    end = start + self.window_length
                    seq = sample_labels[start:end]
                    vals, counts = np.unique(seq[~pd.isna(seq)], return_counts=True)
                    if len(vals) == 0:
                        win_labels.append(0)
                    else:
                        win_labels.append(self.label_map.get(vals[counts.argmax()], 0))
            else:
                # Infer label from filename patterns if requested
                lab = None
                sev = 0
                if self.infer_from_filename:
                    fname = os.path.basename(f).lower()
                    # Patterns: healthy.csv, BearingFault_*.csv, PhaseImbalance_*.csv, RotorFault_*.csv
                    if 'healthy' in fname:
                        lab = 'Healthy'
                        sev = 0
                    elif 'bearingfault' in fname:
                        lab = 'Bearing'
                        # extract severity number if present
                        import re
                        m = re.search(r'bearingfault_(\d+)', fname)
                        if m:
                            sev = int(m.group(1))
                    elif 'phaseimbalance' in fname or 'statorshort' in fname:
                        lab = 'StatorShort'
                        import re
                        m = re.search(r'(phaseimbalance|statorshort)_(\d+)', fname)
                        if m:
                            sev = int(m.group(2))
                    elif 'rotorfault' in fname or 'brokenrotorbar' in fname:
                        lab = 'BrokenRotorBar'
                        import re
                        m = re.search(r'(rotorfault|brokenrotorbar)_(\d+)', fname)
                        if m:
                            sev = int(m.group(2))
                label_int = self.label_map.get(lab or 'Healthy', 0)
                win_labels = [label_int] * n_win

            # Include inferred severity if available
            self.metadata.append({
                "file": f,
                "ch_names": ch_names,
                "aux_cols": aux_cols,
                "n_windows": n_win,
                "is_voltage": is_voltage,
                "severity": locals().get('sev', 0),
            })
            for wi in range(n_win):
                self.index.append((fi, wi))
                self.window_labels.append(win_labels[wi])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fi, wi = self.index[idx]
        meta = self.metadata[fi]
        df = pd.read_csv(self.files[fi])
        ch_names = meta['ch_names']
        core = df[ch_names].to_numpy().T
        aux_arr = []
        for col in meta['aux_cols']:
            aux_arr.append(df[col].to_numpy())
        if aux_arr:
            aux_arr = np.stack(aux_arr, axis=0)
            data = np.concatenate([core, aux_arr], axis=0)
        else:
            data = core
        windows = sliding_windows(data, self.window_length, self.hop_length)
        x = windows[wi]

        if self.scaler is not None:
            if self.scaling_type == 'zscore':
                mean = np.array(self.scaler['mean'])[:, None]
                std = np.array(self.scaler['std'])[:, None]
                x = (x - mean) / (std + 1e-8)
            elif self.scaling_type == 'minmax':
                vmin = np.array(self.scaler['min'])[:, None]
                vmax = np.array(self.scaler['max'])[:, None]
                x = (x - vmin) / (vmax - vmin + 1e-8)
        else:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            x = (x - mean) / (std + 1e-8)

        if self.augment_fn is not None:
            x = self.augment_fn(x)

        label = self.window_labels[idx]
        x = torch.from_numpy(x.astype('float32'))
        if self.return_severity:
            sev = self.metadata[fi].get('severity', 0)
            sev = float(sev) / 6.0  # normalize severity to [0,1]
            return x, int(label), sev
        return x, int(label)


def collate_fn(batch):
    if len(batch[0]) == 3:
        xs, ys, sv = zip(*batch)
        xs = torch.stack(xs, dim=0)
        return xs, torch.tensor(ys, dtype=torch.long), torch.tensor(sv, dtype=torch.float32)
    else:
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=0)
        return xs, torch.tensor(ys, dtype=torch.long)


def compute_scaler(files: Sequence[str], prefer_voltage: bool = False, include_aux: bool = False, scaling_type: str = 'zscore') -> Dict:
    sums = None
    sq_sums = None
    mins = None
    maxs = None
    count = 0
    for f in files:
        df = pd.read_csv(f)
        ch_names, _ = _detect_channels(df, prefer_voltage)
        arrs = [df[c].to_numpy() for c in ch_names]
        if include_aux:
            for col in ["rpm", "RPM", "vibration", "Torque"]:
                if col in df.columns:
                    arrs.append(df[col].to_numpy())
        mat = np.stack(arrs, axis=0)
        if sums is None:
            sums = mat.sum(axis=1)
            sq_sums = (mat ** 2).sum(axis=1)
            mins = mat.min(axis=1)
            maxs = mat.max(axis=1)
        else:
            sums += mat.sum(axis=1)
            sq_sums += (mat ** 2).sum(axis=1)
            mins = np.minimum(mins, mat.min(axis=1))
            maxs = np.maximum(maxs, mat.max(axis=1))
        count += mat.shape[1]
    if scaling_type == 'zscore':
        mean = sums / count
        var = (sq_sums / count) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-12))
        return {'type': 'zscore', 'mean': mean.tolist(), 'std': std.tolist()}
    else:
        return {'type': 'minmax', 'min': mins.tolist(), 'max': maxs.tolist()}


def split_files(all_files: Sequence[str], ratios=(0.7, 0.15, 0.15), seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    files = list(all_files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


def create_datasets(path: str, ratios=(0.7, 0.15, 0.15), seed: int = 42, window_length: int = 3000, hop_length: int = 1500, label_map=None, prefer_voltage=False, file_level_label=False, include_aux=False, scaler: Dict | None = None, scaling_type: str = 'zscore', augment_fn=None, return_severity: bool = False) -> Tuple[MotorCurrentDataset, MotorCurrentDataset, MotorCurrentDataset]:
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.csv')]
    else:
        files = [path]
    train_f, val_f, test_f = split_files(files, ratios, seed)
    if scaler is None:
        scaler = compute_scaler(train_f, prefer_voltage=prefer_voltage, include_aux=include_aux, scaling_type=scaling_type)
    ds_train = MotorCurrentDataset(train_f, window_length, hop_length, label_map, prefer_voltage, file_level_label, scaler, scaling_type, augment_fn, include_aux, return_severity=return_severity)
    ds_val = MotorCurrentDataset(val_f, window_length, hop_length, label_map, prefer_voltage, file_level_label, scaler, scaling_type, None, include_aux, return_severity=return_severity)
    ds_test = MotorCurrentDataset(test_f, window_length, hop_length, label_map, prefer_voltage, file_level_label, scaler, scaling_type, None, include_aux, return_severity=return_severity)
    return ds_train, ds_val, ds_test