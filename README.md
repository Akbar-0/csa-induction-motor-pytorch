# PyTorch 1D-CNN for Motor Current Signature Analysis (MCSA)

This project trains a 1D CNN on three-phase currents to classify motor condition and optionally predict fault severity.

## Environment Setup
1. Clone the repository:
	```bash
	git clone <repository-url>
	cd csa-induction-motor-pytorch
	```

2. Create a Python virtual environment:
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows: venv\Scripts\activate
	```

3. Install dependencies:
	```bash
	pip install --upgrade pip
	pip install -r requirements.txt
	```

4. Verify PyTorch installation:
	```bash
	python -c "import torch; print(torch.__version__)"
	```

5. Prepare data directory:
	```bash
	mkdir -p data/simulink
	```

6. Place your Simulink CSV files in `data/simulink/` following the naming conventions above.

## Simulink CSV Data
- Place your CSVs under `data/simulink`.
- Each CSV contains columns: `Time,Torque,RPM,Ia,Ib,Ic` (order flexible; headers used).
- Filename conventions (case-insensitive):
	- `healthy_1..6.csv` → class `Healthy`, load `1..6`
	- `BearingFault_1..6.csv` → class `Bearing`, severity `1..6`
	- `PhaseImbalance_1..6.csv` or `StatorShort_1..6.csv` → class `StatorShort`, severity `1..6`
	- `RotorFault_1..6.csv` or `BrokenRotorBar_1..6.csv` → class `BrokenRotorBar`, severity `1..6`
- The dataset infers labels and severity from filenames. If no number is present, severity defaults to 0 (ignored in training).

## Training
```powershell
python train.py --config config.yaml
```
___OR if you want customized training:___
```powershell
python "path/to/PyTorch 1DCNN MCSA/train.py" --data-csv "path/to/PyTorch 1DCNN MCSA/data/simulink" --save-dir "path/to/PyTorch 1DCNN MCSA/runs/exp" --epochs 80 --batch-size 32 --severity-head --severity-loss-weight 0.3 --include-aux
```
- `--severity-head`: enables multi-task prediction with a 6-class severity head.
- `--severity-loss-weight`: scales the severity loss term (default `0.3`).
- `--include-aux`: includes `RPM` and `Torque` (and `vibration`/`rpm` when available) as auxiliary channels.
- Defaults like `window_length`, `hop_length`, `scheduler`, etc., are in `config.yaml`.

## Evaluation
```powershell
python "path/to/PyTorch 1DCNN MCSA/evaluate.py" --checkpoint "path/to/PyTorch 1DCNN MCSA/runs/exp/best.pth" --data-csv "d:\VS code\PyTorch 1DCNN MCSA\data\simulink" --save-dir "d:\VS code\PyTorch 1DCNN MCSA\runs\sim_eval"
```
- Saves `confusion_matrix.png`. If severity head is enabled, also saves `severity_hist.png`.

## Prediction
```powershell
python "path/to/PyTorch 1DCNN MCSA/predict.py" --checkpoint "path/to/PyTorch 1DCNN MCSA/runs/exp/best.pth" --data-csv "path/to/PyTorch 1DCNN MCSA/data/simulink" --topk 3
```
- Prints top-k class labels and confidences per window.
- Prints `severity_pred` only when the top predicted class is not `Healthy`.

## Label Mapping
- Classes: `[Healthy, Bearing, BrokenRotorBar, StatorShort]`.
- Severity head: 6 classes representing severity `1..6`. Healthy windows are ignored in severity loss.

## Notes
- Windows are created by sliding over time series (`window_length`, `hop_length` from `config.yaml`).
- Normalization uses z-score by default; scaler computed from training split.
- For Windows, `num_workers` defaults to 0 in notebooks to avoid pickling issues.
