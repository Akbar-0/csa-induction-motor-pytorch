"""Generate synthetic 3-phase current signals for quick testing."""
import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd


def synth_signal(length: int, fs: int, fault: str, rpm: float = 1500.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(length) / fs
    base_freq = rpm / 60.0
    f0 = base_freq

    def phase(A=1.0, phase_shift=0.0, harmonics=None, noise_level=0.01):
        s = A * np.sin(2 * np.pi * f0 * t + phase_shift)
        if harmonics:
            for hf, amp in harmonics:
                s += amp * np.sin(2 * np.pi * hf * f0 * t + phase_shift)
        s += noise_level * np.random.randn(len(t))
        return s

    if fault == 'Healthy':
        Ia = phase(A=1.0, phase_shift=0.0, harmonics=[(3, 0.05)])
        Ib = phase(A=1.0, phase_shift=-2.0 * np.pi / 3, harmonics=[(3, 0.05)])
        Ic = phase(A=1.0, phase_shift=2.0 * np.pi / 3, harmonics=[(3, 0.05)])
    elif fault == 'Bearing':
        Ia = phase(A=1.0, phase_shift=0.0, harmonics=[(1.5, 0.2), (3, 0.05)], noise_level=0.05)
        Ib = phase(A=1.0, phase_shift=-2.0 * np.pi / 3, harmonics=[(1.5, 0.15)], noise_level=0.04)
        Ic = phase(A=1.0, phase_shift=2.0 * np.pi / 3, harmonics=[(1.5, 0.18)], noise_level=0.045)
    elif fault == 'BrokenRotorBar':
        Ia = phase(A=1.2, phase_shift=0.0, harmonics=[(2, 0.3)], noise_level=0.06)
        Ib = phase(A=0.9, phase_shift=-2.0 * np.pi / 3, harmonics=[(2, 0.25)], noise_level=0.05)
        Ic = phase(A=0.95, phase_shift=2.0 * np.pi / 3, harmonics=[(2, 0.22)], noise_level=0.05)
    elif fault == 'StatorShort':
        Ia = phase(A=1.5, phase_shift=0.0, harmonics=[(5, 0.3)], noise_level=0.08)
        Ib = phase(A=1.1, phase_shift=-2.0 * np.pi / 3, harmonics=[(5, 0.25)], noise_level=0.07)
        Ic = phase(A=0.9, phase_shift=2.0 * np.pi / 3, harmonics=[(5, 0.2)], noise_level=0.07)
    else:
        Ia = phase(); Ib = phase(phase_shift=-2.0 * np.pi / 3); Ic = phase(phase_shift=2.0 * np.pi / 3)

    return Ia, Ib, Ic


def make_file(out_path: str, length: int, fs: int, label: str):
    Ia, Ib, Ic = synth_signal(length, fs, label)
    t = np.arange(length) / fs
    df = pd.DataFrame({'time': t, 'Ia': Ia, 'Ib': Ib, 'Ic': Ic, 'label': label})
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--n-files', type=int, default=40)
    parser.add_argument('--length', type=int, default=6000)
    parser.add_argument('--fs', type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    labels = ['Healthy', 'Bearing', 'BrokenRotorBar', 'StatorShort']
    for i in range(args.n_files):
        lab = labels[i % len(labels)]
        fname = os.path.join(args.out_dir, f'synth_{i:04d}_{lab}.csv')
        make_file(fname, args.length, args.fs, lab)
    print('Done. Generated', args.n_files, 'files in', args.out_dir)


if __name__ == '__main__':
    main()