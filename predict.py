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
    
    # Calculate input channels based on auxiliary features
    include_aux = cfg.get('include_aux', False)
    in_channels = 5 if include_aux else 3
    
    model = create_model({
        'in_channels': in_channels, 
        'num_classes': len(classes), 
        'severity_head': cfg.get('severity_head', False), 
        'severity_classes': cfg.get('severity_classes', 7),
        'base_channels': cfg.get('base_channels', 64),
        'dropout': cfg.get('dropout', 0.5)
    })
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    ds = MotorCurrentDataset(
        args.data_csv, 
        window_length=cfg.get('window_length', 3000), 
        hop_length=cfg.get('hop_length', 1500), 
        label_map={c: i for i, c in enumerate(classes)},
        include_aux=include_aux,
        prefer_voltage=cfg.get('prefer_voltage', False),
        scaling_type=cfg.get('scaling_type', 'zscore')
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if cfg.get('severity_head', False):
                x, y, sev = batch
            else:
                x, y = batch
            if args.window_index is not None and i != args.window_index:
                continue
            outputs = model(x)
            if cfg.get('severity_head', False):
                logits, sev_logits = outputs
                sev_probs = torch.softmax(sev_logits, dim=1).cpu().numpy()[0]
                sev_pred_idx = int(np.argmax(sev_probs))
                sev_pred = sev_pred_idx + 1  # map back to 1..6
            else:
                logits = outputs
                sev_pred = None
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            topk_idx = np.argsort(-probs, axis=1)[:, :args.topk]
            pred_labels = [[classes[j] for j in row] for row in topk_idx]
            confidences = [probs[bi, row].tolist() for bi, row in enumerate(topk_idx)]
            res = {'window': i, 'top_labels': pred_labels[0], 'confidences': confidences[0]}
            if sev_pred is not None:
                # Only report severity if top class is not Healthy
                if pred_labels[0][0].lower() != 'healthy':
                    res['severity_pred'] = sev_pred
            results.append(res)
            if args.window_index is not None:
                break
    for r in results:
        sev_txt = f", severity={r['severity_pred']}" if 'severity_pred' in r else ''
        print(f"Window {r['window']}: {list(zip(r['top_labels'], [f'{c:.3f}' for c in r['confidences']]))}{sev_txt}")
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print('Saved predictions to', args.json_out)


if __name__ == '__main__':
    main()