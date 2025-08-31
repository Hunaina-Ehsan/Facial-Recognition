import numpy as np
from utils import _normalize, _cosface_score, _generate_random_planes, _hash_vector, _l2_sq
import yaml
from build_exe import resource_path

config_path = resource_path("config/config.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

NUM_PLANES = cfg["num_planes"]
NUM_TABLES = cfg["num_tables"]

# --- public API ---
def build_lsh_index(embeddings_dict, num_planes=NUM_PLANES, n_tables=NUM_TABLES, seed=42, normalize=True):
    assert len(embeddings_dict) > 0, "embeddings_dict is empty"

    labels = list(embeddings_dict.keys())
    dim = embeddings_dict[labels[0]].shape[0]

    if normalize:
        vectors = {lbl: _normalize(np.asarray(vec, dtype=np.float32))
                   for lbl, vec in embeddings_dict.items()}
    else:
        vectors = {lbl: np.asarray(vec, dtype=np.float32) for lbl, vec in embeddings_dict.items()}

    rng = np.random.RandomState(seed)
    planes_list, buckets_list = [], []

    for t in range(n_tables):
        planes = _generate_random_planes(num_planes, dim, rng)
        planes_list.append(planes)

        buckets = {}
        for lbl, vec in vectors.items():
            h = _hash_vector(vec, planes)
            if h not in buckets:
                buckets[h] = []
            buckets[h].append((lbl, vec))
        buckets_list.append(buckets)

    return planes_list, buckets_list


def _nearby_hashes(h, num_bits, max_flips=2):
    """Generate nearby hash values by flipping up to `max_flips` bits."""
    neighbors = [h]
    for i in range(num_bits):
        neighbors.append(h ^ (1 << i))  # flip one bit
    if max_flips > 1:
        for i in range(num_bits):
            for j in range(i+1, num_bits):
                neighbors.append(h ^ (1 << i) ^ (1 << j))  # flip two bits
    return neighbors


def lsh_lookup(query_vec,
              planes_list,
              buckets_list,
              fallback_embeddings,
              l2_thresh=None,
              probe_all_tables=True,
              normalize=True,
              dyn_thresh=None):
    q = np.asarray(query_vec, dtype=np.float32)
    if normalize:
        q = _normalize(q)

    candidates = {}
    for planes, buckets in zip(planes_list, buckets_list) if probe_all_tables else [(planes_list[0], buckets_list[0])]:
        h = _hash_vector(q, planes)
        # Multi-probe: search nearby buckets (up to 2 bit flips)
        for nh in _nearby_hashes(h, planes.shape[0], max_flips=2):
            for lbl, vec in buckets.get(nh, []):
                if lbl not in candidates:
                    candidates[lbl] = vec

    # If still nothing, fall back to a *small random subset* of embeddings
    if not candidates:
        all_labels = list(fallback_embeddings.keys())
        if len(all_labels) > 50:   # pick subset if DB is large
            sampled = np.random.choice(all_labels, 50, replace=False)
            candidates = {lbl: fallback_embeddings[lbl] for lbl in sampled}
        else:
            candidates = fallback_embeddings

    best_label, best_score = None, -1.0
    for lbl, vec in candidates.items():
        #score = _cosine(q, vec)
        score = _cosface_score(q, vec, margin=0.35, scale=30.0)
        # short-circuit: if score < dyn_thresh, skip
        if dyn_thresh is not None and score < dyn_thresh * 0.5:
            continue
        if l2_thresh is not None:
            if _l2_sq(q, vec) < l2_thresh ** 2 and score > best_score:
                best_label, best_score = lbl, score
        else:
            if score > best_score:
                best_label, best_score = lbl, score

    return best_label if best_label is not None else "Unknown", best_score
