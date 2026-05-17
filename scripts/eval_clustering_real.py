"""Evaluate clustering on SLR80 real embeddings (train split only)."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.clustering.cluster import agg_clustering

EMB_DIR = "data/processed/burmese_asr/train/embeddings"

# 1. Load embeddings
records = []
for npz_path in glob.glob(os.path.join(EMB_DIR, "*.npz")):
    base = os.path.basename(npz_path).replace(".npz", "")
    parts = base.split("_")
    if len(parts) < 3:
        continue
    speaker = parts[1]
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    if emb.ndim == 2:
        emb = emb[0]
    records.append({"emb": emb, "speaker": speaker, "id": base})

all_embs = np.array([r["emb"] for r in records], dtype=np.float32)
all_speakers = [r["speaker"] for r in records]

unique_speakers = sorted(set(all_speakers))
speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}
all_labels = np.array([speaker_to_idx[s] for s in all_speakers])

print(f"Loaded {len(all_embs)} embeddings")
print(f"Ground-truth speakers: {len(unique_speakers)}")

# 2. Intra / inter speaker distance stats
intra_dists = []
inter_dists = []
centroids = []
for spk in unique_speakers:
    mask = np.array([s == spk for s in all_speakers])
    embs = all_embs[mask]
    sims = np.dot(embs, embs.T)
    np.fill_diagonal(sims, 1.0)
    dists = 1 - sims
    intra_dists.extend(dists[np.triu_indices_from(dists, k=1)].tolist())
    c = np.mean(embs, axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    centroids.append(c)

centroids = np.array(centroids)
inter_sims = np.dot(centroids, centroids.T)
np.fill_diagonal(inter_sims, -1)
inter_dists = 1 - inter_sims
inter_dists = inter_dists[inter_dists > 0]

print(f"\nIntra-speaker: mean={np.mean(intra_dists):.3f}, max={np.max(intra_dists):.3f}")
print(f"Inter-speaker: mean={np.mean(inter_dists):.3f}, min={np.min(inter_dists):.3f}")

# 3. Single-stage clustering
print("\n=== Single-stage clustering ===")
for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    labels = agg_clustering(all_embs, distance_threshold=thr)
    n_clusters = len(np.unique(labels))
    purities = []
    for lbl in np.unique(labels):
        gt = all_labels[labels == lbl]
        purities.append(np.bincount(gt).max() / len(gt))
    purity = np.mean(purities)
    print(f"thr={thr:.2f} -> clusters={n_clusters:4d}, purity={purity:.1%}")

# 4. Two-stage clustering
print("\n=== Two-stage clustering ===")
for thr in [0.25, 0.30, 0.35]:
    for cthr in [0.35, 0.40, 0.45, 0.50, 0.55]:
        labels = agg_clustering(
            all_embs,
            distance_threshold=thr,
            centroid_threshold=cthr,
            min_cluster_size=3,
        )
        n_clusters = len(np.unique(labels))
        purities = []
        for lbl in np.unique(labels):
            gt = all_labels[labels == lbl]
            purities.append(np.bincount(gt).max() / len(gt))
        purity = np.mean(purities)
        print(f"thr={thr:.2f}, cthr={cthr:.2f} -> clusters={n_clusters:4d}, purity={purity:.1%}")
