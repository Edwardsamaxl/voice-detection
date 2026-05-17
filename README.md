# Voice Detection

Speaker diarization and voice recognition pipeline.

## Quick Start

Use a project-local virtual environment. Do not install this stack into the
global Python environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchaudio==2.4.1
python -m pip install -r requirements.txt -c constraints-win-cpu.txt
python scripts/doctor.py
python setup_models.py
```

If `doctor.py` reports `HF_TOKEN` missing, set it before running real
diarization:

```powershell
$env:HF_TOKEN = "hf_..."
```

## Project Structure

- `src/` - Source code for audio, diarization, embedding, clustering, and speaker recognition.
- `models/` - Downloaded model weights, ignored by git and populated by `setup_models.py`.
- `data/` - Audio datasets and outputs, ignored by git.
- `tests/` - Test suite.
