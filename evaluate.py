"""CLI to evaluate a checkpoint on a dataset"""
import argparse
import json
import os
from src.eval import evaluate_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-csv', required=True)
    p.add_argument('--save-dir', default='runs/eval')
    p.add_argument('--metrics-json', default='metrics.json')
    return p.parse_args()


def main():
    args = parse_args()
    classes = ["Healthy", "Bearing", "BrokenRotorBar", "StatorShort"]
    metrics = evaluate_checkpoint(args.checkpoint, args.data_csv, classes, save_dir=args.save_dir)
    out_path = os.path.join(args.save_dir, args.metrics_json)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved metrics to', out_path)


if __name__ == '__main__':
    main()