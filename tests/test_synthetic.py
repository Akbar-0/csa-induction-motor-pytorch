"""Minimal tests using synthetic generator to sanity-check dataset and model forward pass."""
import os
import tempfile

import torch

from scripts.generate_synthetic import make_file
from src.data import MotorCurrentDataset
from src.model import Conv1DClassifier


def test_dataset_and_model():
    tmp = tempfile.mkdtemp(prefix='mcsa_test_')
    os.makedirs(tmp, exist_ok=True)
    for i, lab in enumerate(['Healthy', 'Bearing', 'BrokenRotorBar']):
        make_file(os.path.join(tmp, f'test_{i}_{lab}.csv'), length=3000, fs=1000, label=lab)

    ds = MotorCurrentDataset(tmp, window_length=1024, hop_length=512)
    x, y = ds[0]
    assert x.shape[0] == 3
    model = Conv1DClassifier(in_channels=3, num_classes=4)
    out = model(x.unsqueeze(0))
    assert out.shape[0] == 1 and out.shape[1] == 4


if __name__ == '__main__':
    test_dataset_and_model()
    print('Sanity test passed')