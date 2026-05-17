"""Build the speaker repository from cached embeddings.

Usage:
    python scripts/build_speaker_repository.py \
        --embedding_dir data/processed/burmese_asr/embeddings \
        --output_dir data/processed/speaker_db
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CENTROID_THRESHOLD, DISTANCE_THRESHOLD, MIN_CLUSTER_SIZE
from src.clustering.cluster import (
    assign_global_speakers,
    assign_global_speakers_from_slr80_filename,
)
from src.clustering.pool import build_embedding_pool
from src.core.repository import SpeakerRepository
from src.core.storage import JsonStorage, PickleStorage
from src.speaker_db.vector_index import VectorIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build speaker repository from embeddings.")
    parser.add_argument(
        "--embedding_dir",
        default="data/processed/burmese_asr/embeddings",
        help="Directory containing .npz embedding files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/speaker_db",
        help="Directory to save speaker repository.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=DISTANCE_THRESHOLD,
        help="Clustering distance threshold.",
    )
    parser.add_argument(
        "--centroid_threshold",
        type=float,
        default=CENTROID_THRESHOLD,
        help="Second-stage centroid merge threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=MIN_CLUSTER_SIZE,
        help="Minimum cluster size for second-stage merging.",
    )
    parser.add_argument(
        "--label_strategy",
        default="cluster",
        choices=["cluster", "slr80_filename"],
        help="How to assign global speaker labels before building the repository.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading embeddings from %s...", args.embedding_dir)
    pool = build_embedding_pool(args.embedding_dir)
    logger.info("Loaded %d segments with embeddings.", len(pool))

    if len(pool) == 0:
        logger.error("No segments found. Abort.")
        sys.exit(1)

    if args.label_strategy == "cluster":
        logger.info("Clustering with distance_threshold=%s...", args.distance_threshold)
        clustered_pool = assign_global_speakers(
            pool,
            distance_threshold=args.distance_threshold,
            centroid_threshold=args.centroid_threshold,
            min_cluster_size=args.min_cluster_size,
        )
    else:
        logger.info("Assigning SLR80 speaker ids from filenames...")
        clustered_pool = assign_global_speakers_from_slr80_filename(pool)
    logger.info("Assigned %d global speakers.", len(clustered_pool.global_speakers()))

    logger.info("Building speaker repository...")
    vector_index = VectorIndex()
    repo = SpeakerRepository(
        vector_index=vector_index,
        storage=JsonStorage(args.output_dir),
    )
    repo.build_from_pool(clustered_pool)
    logger.info("Built speaker repository with %d speakers.", len(repo.all_speakers()))

    logger.info("Persisting to %s...", args.output_dir)
    repo.save("speaker_db:main")

    pickle_storage = PickleStorage(args.output_dir)
    pickle_storage.save(
        "vector_index:main",
        {
            "vectors": vector_index.vectors,
            "labels": vector_index.labels,
        },
    )

    logger.info(
        "FAISS index: %d vectors, %d unique speakers.",
        len(vector_index.labels),
        len(set(vector_index.labels)),
    )
    logger.info("Done. Repository saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
