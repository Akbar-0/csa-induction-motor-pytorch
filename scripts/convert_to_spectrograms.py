"""Convert segmented windows to spectrogram-style numpy arrays for 2D CNN experiments."""
import argparse
import os
from pathlib import Path

import numpy as np
from scipy.signal import spectrogram

from src.data import MotorCurrentDataset


def window_to_spec(window: np.ndarray, fs: int = 1000, nperseg: int = 256):
    specs = []
    for c in range(window.shape[0]):
        f, t, Sxx = spectrogram(window[c], fs=fs, nperseg=nperseg)
        specs.append(np.log1p(Sxx))
    return np.stack(specs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-csv', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--fs', type=int, default=1000)
    p.add_argument('--nperseg', type=int, default=256)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = MotorCurrentDataset(args.data_csv)
    for i in range(len(ds)):
        x, y = ds[i]
        spec = window_to_spec(x.numpy(), fs=args.fs, nperseg=args.nperseg)
        np.save(Path(args.out_dir) / f'win_{i:05d}_lab{y}.npy', spec)
    print('Saved spectrograms to', args.out_dir)


if __name__ == '__main__':
    main()