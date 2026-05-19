"""Global configuration for voice-detection pipeline."""

import os

from dotenv import load_dotenv

load_dotenv()

# Audio processing
SAMPLE_RATE = 16000
SAMPLE_FMT = "s16"
CHANNELS = 1
TARGET_RMS = 0.1

# Segment filtering
MIN_DURATION = 1.0
MIN_DURATION_ON = 0.3
MIN_DURATION_OFF = 0.2

# Embedding
EMBEDDING_DIM = 192

# Clustering
DISTANCE_THRESHOLD = 0.3
CENTROID_THRESHOLD = 0.35
DEDUP_THRESHOLD = 0.95

# Speaker DB
MAX_EMB = 20

# Recognition
TOPK = 5
T_LOW = 0.50
T_HIGH = 0.65

# ASR / Alignment
MIN_ASR_SEGMENT = 0.5

# Dynamic update
UPDATE_MIN_DURATION = 2.0
UPDATE_DEDUP_THRESHOLD = 0.95

# Translation
TRANSLATION_MODEL = os.environ.get(
    "TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M"
)
TRANSLATION_SRC_LANG = "mya_Mymr"
TRANSLATION_TGT_LANG = "zho_Hans"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
UNIASR_MODEL_DIR = os.path.join(MODELS_DIR, "uni_asr")
UNIASR_MODEL_ID = "iic/speech_UniASR_asr_2pass-my-16k-common-vocab696-pytorch"
