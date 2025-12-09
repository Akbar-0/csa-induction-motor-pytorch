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
│  ├─ simulink
│  ├─ synthetic           # created by in-training synthetic generation
│  └─ nb_quick/           # created by notebook synthetic generation
```

## File Descriptions

- `requirements.txt`: Contains the list of Python packages required for the project. Use `pip install -r requirements.txt` to install dependencies.
- `config.yaml`: Configuration file for setting parameters and defaults for training and evaluation. Modify this file to customize your training setup.
- `train.py`: Main script for training the 1D CNN model on the dataset. It handles data loading, model training, and saving checkpoints.
- `evaluate.py`: Script for evaluating the trained model on a test dataset. It generates performance metrics and visualizations.
- `predict.py`: Script for making predictions using the trained model. It takes input data and outputs predictions.
- `src/`: Directory containing the source code for the project.
    - `__init__.py`: Initializes the `src` package.
    - `data.py`: Contains functions for loading and preprocessing data.
    - `model.py`: Defines the architecture of the 1D CNN model.
    - `train.py`: Contains functions specific to the training process, including loss calculation and optimization.
    - `eval.py`: Contains functions for evaluating the model's performance.
    - `utils.py`: Utility functions used throughout the project, such as logging and visualization helpers.
- `scripts/`: Directory for utility scripts.
    - `generate_synthetic.py`: Script for generating synthetic data for training and testing.
    - `convert_to_spectrograms.py`: Script for converting time-series data into spectrograms for analysis.
- `tests/`: Directory containing unit tests for the project.
    - `test_synthetic.py`: Tests for the synthetic data generation functions.
- `notebook/`: Directory containing Jupyter notebooks for experimentation and quick demos.
    - `quickstart.ipynb`: A notebook providing a quick start guide to using the project.
- `runs/`: Directory for storing outputs from training runs, including checkpoints and logs.
    - `nb_demo/`: Created by notebook training, contains checkpoints and logs.
- `data/`: Directory for storing datasets.
    - `nb_quick/`: Created by notebook synthetic generation, contains generated data files.

## Notes:
- `runs/` and `data/` entries are created at runtime by the notebook/train scripts.
- Edit `config.yaml` to change defaults; notebook overrides some keys for quick demos.
- Checkpoints and evaluation outputs go under the configured `save_dir` (e.g., `runs/nb_demo`).