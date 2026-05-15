"""Build the global embedding pool from a dataset of wav files.

This script runs the full pipeline (diarization -> postprocess -> embedding extraction)
on all wav files in the input directory and saves the results to the output directory.
It supports resume: already-processed files are skipped.

Usage:
    python scripts/build_embedding_pool.py \
        --input_dir data/processed/burmese_asr/wav \
        --output_dir data/processed/burmese_asr/embeddings \
        --segment_dir data/processed/burmese_asr/segments
"""

import argparse
import logging
import os
import sys
import time
import warnings
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MIN_DURATION, MIN_DURATION_OFF, MIN_DURATION_ON, SAMPLE_RATE
from src.audio.preprocess import convert_to_wav, load_wav
from src.diarization.cache import load_segments, save_segments
from src.diarization.postprocess import annotation_to_segments, merge_short_segments
from src.diarization.segment import run_diarization
from src.embedding.cache import save_embeddings
from src.embedding.extractor import EmbeddingExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_file(
    wav_path: str,
    output_dir: str,
    segment_dir: str,
    extractor: EmbeddingExtractor,
    min_duration: float = MIN_DURATION,
    min_duration_on: float = MIN_DURATION_ON,
    min_duration_off: float = MIN_DURATION_OFF,
    skip_existing: bool = True,
    hf_token: str | None = None,
) -> dict:
    """Process a single wav file through the full pipeline.

    Returns:
        A dict with keys: file, status, segment_count, error (optional).
    """
    basename = Path(wav_path).stem
    embedding_path = os.path.join(output_dir, f"{basename}.npz")
    segment_path = os.path.join(segment_dir, f"{basename}.json")

    if skip_existing and os.path.exists(embedding_path):
        return {"file": wav_path, "status": "skipped", "segment_count": 0}

    try:
        # Step 1: Load or run diarization
        if os.path.exists(segment_path):
            logger.info(f"[{basename}] Loading cached diarization...")
            segments = load_segments(segment_path)
        else:
            logger.info(f"[{basename}] Running diarization...")
            annotation = run_diarization(wav_path, token=hf_token)
            merged = merge_short_segments(
                annotation,
                min_duration_on=min_duration_on,
                min_duration_off=min_duration_off,
            )
            segments = annotation_to_segments(merged, min_duration=min_duration)
            save_segments(segment_path, segments)

        if not segments:
            logger.warning(f"[{basename}] No segments found after filtering.")
            return {"file": wav_path, "status": "empty", "segment_count": 0}

        # Enrich segments with file key and segment_id for global pooling
        for i, seg in enumerate(segments):
            seg["file"] = basename
            seg["segment_id"] = f"{basename}_{i:04d}"

        # Step 2: Extract embeddings
        logger.info(f"[{basename}] Extracting embeddings for {len(segments)} segments...")
        segments_with_emb = extractor.extract_segments(wav_path, segments)

        # Filter out segments where embedding failed
        valid_segments = [s for s in segments_with_emb if s.get("embedding") is not None]
        failed_count = len(segments_with_emb) - len(valid_segments)
        if failed_count > 0:
            logger.warning(f"[{basename}] {failed_count} segments failed embedding extraction.")

        if not valid_segments:
            return {"file": wav_path, "status": "empty", "segment_count": 0}

        # Step 3: Save embeddings
        save_embeddings(embedding_path, valid_segments)

        return {
            "file": wav_path,
            "status": "success",
            "segment_count": len(valid_segments),
        }

    except Exception as e:
        logger.error(f"[{basename}] Failed: {e}")
        return {"file": wav_path, "status": "error", "segment_count": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Build embedding pool from a directory of wav files."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing wav files to process.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save embedding npz files.",
    )
    parser.add_argument(
        "--segment_dir",
        default=None,
        help="Directory to cache diarization segments. Defaults to <output_dir>/../segments.",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.wav",
        help="Glob pattern to match wav files under input_dir.",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=MIN_DURATION,
        help=f"Minimum segment duration in seconds (default: {MIN_DURATION}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for models ('cpu', 'cuda', or auto-detect).",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Re-process files even if embedding cache exists.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace token for gated models (pyannote).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    segment_dir = args.segment_dir or os.path.join(
        os.path.dirname(args.output_dir), "segments"
    )
    os.makedirs(segment_dir, exist_ok=True)

    pattern = os.path.join(args.input_dir, args.pattern)
    wav_files = sorted(glob(pattern, recursive=True))

    if not wav_files:
        logger.error(f"No wav files found matching: {pattern}")
        sys.exit(1)

    logger.info(f"Found {len(wav_files)} wav files. Output: {args.output_dir}")
    logger.info("Initializing embedding extractor...")
    extractor = EmbeddingExtractor(device=args.device)

    results = []
    skipped = 0
    success = 0
    empty = 0
    errors = 0

    start_time = time.time()
    for i, wav_path in enumerate(wav_files, 1):
        logger.info(f"[{i}/{len(wav_files)}] Processing: {wav_path}")
        result = process_file(
            wav_path=wav_path,
            output_dir=args.output_dir,
            segment_dir=segment_dir,
            extractor=extractor,
            min_duration=args.min_duration,
            skip_existing=not args.no_skip,
            hf_token=args.hf_token,
        )
        results.append(result)

        status = result["status"]
        if status == "success":
            success += 1
        elif status == "skipped":
            skipped += 1
        elif status == "empty":
            empty += 1
        else:
            errors += 1

    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"Done. Total: {len(wav_files)} | Success: {success} | Skipped: {skipped} | Empty: {empty} | Errors: {errors}")
    logger.info(f"Elapsed time: {elapsed:.1f}s ({elapsed / len(wav_files):.1f}s per file)")

    if errors > 0:
        logger.info("Failed files:")
        for r in results:
            if r["status"] == "error":
                logger.info(f"  - {r['file']}: {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
