# Causal-Inference-for-Health-Policy

## Setup

Create virtual environment and activate:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

Run tests:
```powershell
python test_meta_dml.py
```

If all tests pass, run main analysis:
```powershell
python main.py
```
Press `3` when prompted.

Generate visualizations:
```powershell
python extract_cikm_findings.py
```
