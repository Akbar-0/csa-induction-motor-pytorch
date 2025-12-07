# 1:
<code> Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process </code>

# 2:
<code> & "D:/VS code/PyTorch 1DCNN MCSA/.venv/Scripts/Activate.ps1" </code>

# 3:
<code> .\.venv\Scripts\python.exe -m pip install -r ".\requirements.txt" </code>

# 4:
<code> .\.venv\Scripts\python.exe -m pip install ipykernel </code>

# 5:
<code> .\.venv\Scripts\python.exe -m ipykernel install --user --name mcsa-venv --display-name "Python 3.10 (mcsa .venv)" </code>

# 6:
<code> python "d:\VS code\PyTorch 1DCNN MCSA\train.py" --data-csv "d:\VS code\PyTorch 1DCNN MCSA\data\synthetic" --epochs 2 --batch-size 8 --save-dir "d:\VS code\PyTorch 1DCNN MCSA\runs\demo" --augment </code>

# 7:
<code> .\.venv\Scripts\python.exe ".\train.py" --data-csv ".\data\synthetic" --epochs 2 --batch-size 8 --save-dir ".\runs\demo" --augment </code>

# 8:
<code> python "d:\VS code\PyTorch 1DCNN MCSA\evaluate.py" --checkpoint "d:\VS code\PyTorch 1DCNN MCSA\runs\demo\best.pth" --data-csv "d:\VS code\PyTorch 1DCNN MCSA\data\synthetic" --save-dir "d:\VS code\PyTorch 1DCNN MCSA\runs\demo\eval" </code>