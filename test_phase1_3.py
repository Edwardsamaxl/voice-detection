"""快速验证 Phase 1-3 是否打通。只跑 1 个文件。"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SAMPLE_RATE
from src.audio.preprocess import convert_to_wav, load_wav, crop_segment, rms_normalize
from src.diarization.segment import run_diarization
from src.diarization.postprocess import merge_short_segments, annotation_to_segments
from src.embedding.extractor import EmbeddingExtractor

INPUT_WAV = "data/raw/burmese_asr/audio/bur_0366_0045318711.wav"
OUTPUT_DIR = "data/processed/burmese_asr/test_output"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def phase1_preprocess(input_path: str) -> str:
    """Phase 1: 音频标准化。"""
    print("\n=== Phase 1: Audio Preprocess ===")
    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{basename}_16k.wav")

    if not os.path.exists(output_path):
        convert_to_wav(input_path, output_path)
        print(f"Converted -> {output_path}")
    else:
        print(f"Reuse cached: {output_path}")

    wav, sr = load_wav(output_path)
    print(f"Loaded: shape={wav.shape}, sr={sr}, dtype={wav.dtype}")
    assert sr == SAMPLE_RATE, f"Expected sr={SAMPLE_RATE}, got {sr}"
    return output_path


def phase2_diarization(wav_path: str) -> list:
    """Phase 2: 说话人分段。"""
    print("\n=== Phase 2: Diarization ===")
    print(f"HF_TOKEN present: {HF_TOKEN is not None}")

    annotation = run_diarization(wav_path, token=HF_TOKEN)
    merged = merge_short_segments(annotation)
    segments = annotation_to_segments(merged, min_duration=1.0)
    print(f"Segments found: {len(segments)}")
    for seg in segments[:3]:
        print(f"  {seg.start:.2f}s - {seg.end:.2f}s | speaker={seg.local_speaker}")
    if len(segments) > 3:
        print(f"  ... and {len(segments) - 3} more")
    return segments


def phase3_embedding(wav_path: str, segments: list) -> list:
    """Phase 3: 声纹提取。"""
    print("\n=== Phase 3: Embedding Extraction ===")
    extractor = EmbeddingExtractor(device="cpu")
    print(f"Extractor initialized on {extractor.device}")

    start = time.time()
    enriched = extractor.extract_segments(wav_path, segments)
    elapsed = time.time() - start

    valid = [s for s in enriched if s.embedding is not None]
    print(f"Extracted: {len(valid)}/{len(enriched)} segments in {elapsed:.2f}s")

    if valid:
        emb = valid[0].embedding
        print(f"Embedding dim: {emb.shape}, dtype: {emb.dtype}")
        print(f"Embedding norm (L2): {float((emb ** 2).sum() ** 0.5):.6f}")
    return enriched


def main():
    print(f"Test file: {INPUT_WAV}")
    assert os.path.exists(INPUT_WAV), f"File not found: {INPUT_WAV}"

    try:
        wav_path = phase1_preprocess(INPUT_WAV)
    except Exception as e:
        print(f"[FAIL] Phase 1: {e}")
        sys.exit(1)

    try:
        segments = phase2_diarization(wav_path)
    except Exception as e:
        print(f"[FAIL] Phase 2: {e}")
        sys.exit(1)

    try:
        enriched = phase3_embedding(wav_path, segments)
    except Exception as e:
        print(f"[FAIL] Phase 3: {e}")
        sys.exit(1)

    print("\n=== ALL PHASES PASSED ===")


if __name__ == "__main__":
    main()
