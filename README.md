# Voice Detection

Speaker diarization and voice recognition pipeline.

## Quick Start

```bash
pip install -r requirements.txt
python setup_models.py
```

## Project Structure

- `src/` — Source code (audio, diarization, embedding, clustering, etc.)
- `models/` — Downloaded model weights (ignored by git, populated by `setup_models.py`)
- `data/` — Audio datasets and outputs (ignored by git)
- `tests/` — Test suite
