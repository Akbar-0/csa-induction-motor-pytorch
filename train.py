"""CLI entrypoint for training"""
import argparse
import yaml
from src.train import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--data-csv', type=str, required=True, help='CSV directory or single file')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--save-dir', type=str, default=None)
    p.add_argument('--window-len', type=int, default=None)
    p.add_argument('--hop-len', type=int, default=None)
    p.add_argument('--use-gpu', action='store_true')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--scheduler', type=str, choices=['plateau','cosine'], default=None)
    p.add_argument('--mixed-precision', action='store_true')
    p.add_argument('--early-stop', type=int, default=None)
    p.add_argument('--oversample', action='store_true')
    p.add_argument('--prefer-voltage', action='store_true')
    p.add_argument('--include-aux', action='store_true')
    p.add_argument('--scaling-type', type=str, choices=['zscore','minmax'], default=None)
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f).get('defaults', {})
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.lr:
        cfg['lr'] = args.lr
    if args.save_dir:
        cfg['save_dir'] = args.save_dir
    if args.window_len:
        cfg['window_length'] = args.window_len
    if args.hop_len:
        cfg['hop_length'] = args.hop_len
    if args.use_gpu:
        cfg['use_gpu'] = True
    if args.augment:
        cfg.setdefault('augment', {})['enabled'] = True
    if args.scheduler:
        cfg['scheduler'] = args.scheduler
    if args.mixed_precision:
        cfg['mixed_precision'] = True
    if args.early_stop:
        cfg['early_stop_patience'] = args.early_stop
    if args.oversample:
        cfg['oversample'] = True
    if args.prefer_voltage:
        cfg['prefer_voltage'] = True
    if args.include_aux:
        cfg['include_aux'] = True
    if args.scaling_type:
        cfg['scaling_type'] = args.scaling_type
    if args.resume:
        cfg['resume_path'] = args.resume

    trainer = Trainer(cfg, args.data_csv)
    trainer.fit()


if __name__ == '__main__':
    main()