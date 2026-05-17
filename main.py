"""CLI entry point for voice-detection pipeline.

Subcommands:
    build       Build speaker repository from cached embeddings.
    recognize   Run end-to-end recognition on a new audio file.
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED_DIR


def cmd_build(args: argparse.Namespace) -> None:
    """Build speaker repository from embeddings."""
    from src.core.repository import SpeakerRepository
    from src.core.storage import JsonStorage, PickleStorage
    from src.speaker_db.vector_index import FaissVectorIndex
    from pipeline import build_pipeline

    embedding_dir = args.embedding_dir or os.path.join(PROCESSED_DIR, "burmese_asr", "embeddings")
    output_dir = args.output_dir or os.path.join(PROCESSED_DIR, "speaker_db")

    vector_index = FaissVectorIndex()
    repo = SpeakerRepository(vector_index=vector_index, storage=JsonStorage(output_dir))
    repo = build_pipeline(embedding_dir, label_strategy=args.label_strategy, repo=repo)

    os.makedirs(output_dir, exist_ok=True)
    repo.save("speaker_db:main")

    pickle_storage = PickleStorage(output_dir)
    pickle_storage.save(
        "vector_index:main",
        {
            "vectors": vector_index.vectors,
            "labels": vector_index.labels,
        },
    )

    print(f"Built repository with {len(repo.all_speakers())} speakers at {output_dir}")


def cmd_recognize(args: argparse.Namespace) -> None:
    """Recognize speakers in a new audio file."""
    from src.audio.preprocess import convert_to_wav
    from src.core.repository import SpeakerRepository
    from src.core.storage import JsonStorage, PickleStorage
    from src.diarization.segment import _load_pipeline_class

    # pyannote must be imported before SpeechBrain on this Windows CPU stack,
    # otherwise SpeechBrain lazy imports can trip pyannote/torchvision loading.
    _load_pipeline_class()

    from src.embedding.extractor import EmbeddingExtractor
    from src.speaker_db.vector_index import FaissVectorIndex
    from src.translation.translator import Translator
    from pipeline import recognize_pipeline

    repo_dir = args.repo_dir or os.path.join(PROCESSED_DIR, "speaker_db")

    vector_index = FaissVectorIndex()
    repo = SpeakerRepository(vector_index=vector_index, storage=JsonStorage(repo_dir))
    repo.load("speaker_db:main")

    pickle_storage = PickleStorage(repo_dir)
    if pickle_storage.exists("vector_index:main"):
        data = pickle_storage.load("vector_index:main")
        vector_index.build(data["vectors"], data["labels"])
    else:
        repo.rebuild()

    input_path = args.input
    wav_path = input_path
    temp_wav = None
    if not input_path.lower().endswith(".wav"):
        temp_wav = tempfile.mktemp(suffix=".wav")
        convert_to_wav(input_path, temp_wav)
        wav_path = temp_wav

    try:
        extractor = EmbeddingExtractor(device=args.device)
        translator = Translator(device=args.device) if not args.no_translate else None

        segments = recognize_pipeline(
            wav_path=wav_path,
            repo=repo,
            extractor=extractor,
            translator=translator,
            asr_model=args.asr_model,
            hf_token=args.hf_token,
            device=args.device,
        )

        result = [
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "speaker": seg.global_speaker or seg.local_speaker or "UNKNOWN",
                "display_name": seg.display_name,
                "text": seg.text,
                "translation": seg.translation,
            }
            for seg in segments
        ]

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Result saved to {args.output}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice detection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build speaker repository from embeddings")
    build_parser.add_argument("--embedding_dir", default=None, help="Directory with .npz embedding files")
    build_parser.add_argument("--output_dir", default=None, help="Output directory for speaker_db")
    build_parser.add_argument(
        "--label_strategy",
        default="cluster",
        choices=["cluster", "slr80_filename"],
        help="How to assign global speaker labels before building the repository",
    )

    rec_parser = subparsers.add_parser("recognize", help="Recognize speakers in a new audio file")
    rec_parser.add_argument("input", help="Input audio file path")
    rec_parser.add_argument("--repo_dir", default=None, help="Speaker repository directory")
    rec_parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    rec_parser.add_argument("--device", default=None, help="Device for models (cpu/cuda)")
    rec_parser.add_argument("--asr_model", default="base", help="Whisper model name")
    rec_parser.add_argument("--hf_token", default=None, help="HuggingFace token for gated models")
    rec_parser.add_argument("--no_translate", action="store_true", help="Skip translation")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "recognize":
        cmd_recognize(args)


if __name__ == "__main__":
    main()
