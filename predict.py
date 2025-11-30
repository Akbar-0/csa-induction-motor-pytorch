"""Simple predictor CLI for one CSV or directory"""
import argparse
import os
import json
import numpy as np
import torch

from src.model import create_model
from src.data import MotorCurrentDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-csv', required=True)
    p.add_argument('--topk', type=int, default=1)
    p.add_argument('--window-index', type=int, default=None, help='Optional single window index to predict')
    p.add_argument('--json-out', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt.get('cfg', {})
    classes = cfg.get('classes', ["Healthy", "Bearing", "BrokenRotorBar", "StatorShort"])
    model = create_model({'in_channels': 3, 'num_classes': len(classes)})
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    ds = MotorCurrentDataset(args.data_csv, window_length=cfg.get('window_length', 3000), hop_length=cfg.get('hop_length', 1500), label_map={c: i for i, c in enumerate(classes)})
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    results = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if args.window_index is not None and i != args.window_index:
                continue
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            topk_idx = np.argsort(-probs, axis=1)[:, :args.topk]
            pred_labels = [[classes[j] for j in row] for row in topk_idx]
            confidences = [probs[bi, row].tolist() for bi, row in enumerate(topk_idx)]
            results.append({'window': i, 'top_labels': pred_labels[0], 'confidences': confidences[0]})
            if args.window_index is not None:
                break
    for r in results:
        print(f"Window {r['window']}: {list(zip(r['top_labels'], [f'{c:.3f}' for c in r['confidences']]))}")
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print('Saved predictions to', args.json_out)


if __name__ == '__main__':
    main()