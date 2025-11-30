# Project Structure: PyTorch 1DCNN MCSA

```
PyTorch 1DCNN MCSA/
├─ requirements.txt
├─ config.yaml
├─ PROJECT_STRUCTURE.md
├─ train.py
├─ evaluate.py
├─ predict.py
├─ src/
│  ├─ __init__.py
│  ├─ data.py
│  ├─ model.py
│  ├─ train.py
│  ├─ eval.py
│  ├─ utils.py
├─ scripts/
│  ├─ generate_synthetic.py
│  ├─ convert_to_spectrograms.py
├─ tests/
│  ├─ test_synthetic.py
├─ notebook/
│  ├─ quickstart.ipynb
├─ runs/
│  └─ nb_demo/            # created by notebook training (checkpoints, logs)
├─ data/
│  └─ nb_quick/           # created by notebook synthetic generation
```

Notes:
- `runs/` and `data/` entries are created at runtime by the notebook/train scripts.
- Edit `config.yaml` to change defaults; notebook overrides some keys for quick demos.
- Checkpoints and evaluation outputs go under the configured `save_dir` (e.g., `runs/nb_demo`).
