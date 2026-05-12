import numpy as np
from scipy.spatial.distance import pdist
import os
import pickle
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import pairwise_distances

def estimate_typical_distance(X, sample_pairs=50000, seed=42):
    """
    Estimate the median pairwise Euclidean distance among table embeddings.

    Parameters
    ----------
    X : np.ndarray of shape (n_tables, d)
        Clean table embeddings.
    sample_pairs : int or None
        If None or if n_tables is small, compute exact median over all pairs.
        Otherwise, estimate it from randomly sampled pairs.
    seed : int
        Random seed for pair sampling.

    Returns
    -------
    m : float
        Estimated median pairwise Euclidean distance.
    """
    n, d = X.shape

    # exact computation for relatively small datasets
    if sample_pairs is None or n <= 2000:
        distances = pdist(X, metric="euclidean")
        return float(np.median(distances))

    # approximate computation by random pair sampling
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=sample_pairs)
    j = rng.integers(0, n, size=sample_pairs)

    mask = i != j
    i = i[mask]
    j = j[mask]

    distances = np.linalg.norm(X[i] - X[j], axis=1)
    return float(np.median(distances))


def add_gaussian_noise_to_table_embeddings(
    X,
    noise_ratio,
    pairwise_median=None,
    sample_pairs=50000,
    seed=42,
    renormalize=False
):
    """
    Add isotropic Gaussian noise to final table embeddings.

    Parameters
    ----------
    X : np.ndarray of shape (n_tables, d)
        Clean table embeddings.
    noise_ratio : float
        Lambda in the paper, e.g. 0.02 / 0.05 / 0.10.
    pairwise_median : float or None
        Precomputed median pairwise distance m. If None, estimate it from X.
    sample_pairs : int
        Number of sampled pairs for estimating m when needed.
    seed : int
        Random seed for Gaussian noise.
    renormalize : bool
        Whether to L2-normalize embeddings after noise injection.
        Set this to True only if your original pipeline uses normalized table embeddings.

    Returns
    -------
    X_noisy : np.ndarray
        Noisy table embeddings.
    sigma : float
        Standard deviation used for each dimension.
    m : float
        Median pairwise distance used for calibration.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    if pairwise_median is None:
        m = estimate_typical_distance(X, sample_pairs=sample_pairs, seed=seed)
    else:
        m = float(pairwise_median)

    sigma = (noise_ratio * m) / np.sqrt(2 * d)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    X_noisy = X + noise

    if renormalize:
        norms = np.linalg.norm(X_noisy, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        X_noisy = X_noisy / norms

    return X_noisy, sigma, m

def sample_pairwise_distances(X, sample_pairs=50000, seed=42):
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=sample_pairs)
    j = rng.integers(0, n, size=sample_pairs)

    mask = i != j
    i = i[mask]
    j = j[mask]

    distances = np.linalg.norm(X[i] - X[j], axis=1)
    return distances, i, j

def pairwise_distance_sanity_check(X_clean, X_noisy, sample_pairs=50000, seed=42):
    X_clean = np.asarray(X_clean, dtype=np.float64)
    X_noisy = np.asarray(X_noisy, dtype=np.float64)

    n, d = X_clean.shape

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=sample_pairs)
    j = rng.integers(0, n, size=sample_pairs)

    mask = i != j
    i = i[mask]
    j = j[mask]

    clean_dist = np.linalg.norm(X_clean[i] - X_clean[j], axis=1)
    noisy_dist = np.linalg.norm(X_noisy[i] - X_noisy[j], axis=1)

    abs_change = noisy_dist - clean_dist
    rel_change = abs_change / np.clip(clean_dist, 1e-12, None)

    print("Clean pairwise distance:")
    print(np.percentile(clean_dist, [0, 10, 25, 50, 75, 90, 100]))

    print("\nNoisy pairwise distance:")
    print(np.percentile(noisy_dist, [0, 10, 25, 50, 75, 90, 100]))

    print("\nAbsolute change: noisy - clean")
    print(np.percentile(abs_change, [0, 10, 25, 50, 75, 90, 100]))

    print("\nRelative change: (noisy - clean) / clean")
    print(np.percentile(rel_change, [0, 10, 25, 50, 75, 90, 100]))

    print("\nMedian clean distance:", np.median(clean_dist))
    print("Median noisy distance:", np.median(noisy_dist))
    print("Median absolute change:", np.median(abs_change))
    print("Median relative change:", np.median(rel_change))

    return {
        "clean_dist": clean_dist,
        "noisy_dist": noisy_dist,
        "abs_change": abs_change,
        "rel_change": rel_change,
    }

def perturbation_scale_check(X_clean, X_noisy, noise_ratio, m, sample_pairs=50000, seed=42):
    X_clean = np.asarray(X_clean, dtype=np.float64)
    X_noisy = np.asarray(X_noisy, dtype=np.float64)

    noise = X_noisy - X_clean

    n, d = X_clean.shape
    rng = np.random.default_rng(seed)

    i = rng.integers(0, n, size=sample_pairs)
    j = rng.integers(0, n, size=sample_pairs)

    mask = i != j
    i = i[mask]
    j = j[mask]

    pairwise_noise_diff = noise[i] - noise[j]
    pairwise_noise_norm = np.linalg.norm(pairwise_noise_diff, axis=1)

    empirical_rms = np.sqrt(np.mean(pairwise_noise_norm ** 2))
    target_rms = noise_ratio * m

    print(f"Target pairwise RMS perturbation = {target_rms:.6f}")
    print(f"Empirical pairwise RMS perturbation = {empirical_rms:.6f}")
    print(f"Ratio empirical / target = {empirical_rms / target_rms:.4f}")

    return empirical_rms


def pairwise_rank_correlation_check(X_clean, X_noisy, sample_pairs=50000, seed=42):
    X_clean = np.asarray(X_clean, dtype=np.float64)
    X_noisy = np.asarray(X_noisy, dtype=np.float64)

    n, d = X_clean.shape
    rng = np.random.default_rng(seed)

    i = rng.integers(0, n, size=sample_pairs)
    j = rng.integers(0, n, size=sample_pairs)

    mask = i != j
    i = i[mask]
    j = j[mask]

    clean_dist = np.linalg.norm(X_clean[i] - X_clean[j], axis=1)
    noisy_dist = np.linalg.norm(X_noisy[i] - X_noisy[j], axis=1)

    pearson_corr = pearsonr(clean_dist, noisy_dist)[0]
    spearman_corr = spearmanr(clean_dist, noisy_dist)[0]

    print(f"Pearson correlation of pairwise distances = {pearson_corr:.4f}")
    print(f"Spearman rank correlation of pairwise distances = {spearman_corr:.4f}")

    return pearson_corr, spearman_corr


def nearest_neighbor_overlap_check(X_clean, X_noisy, k=10):
    D_clean = pairwise_distances(X_clean, metric="euclidean")
    D_noisy = pairwise_distances(X_noisy, metric="euclidean")

    np.fill_diagonal(D_clean, np.inf)
    np.fill_diagonal(D_noisy, np.inf)

    nn_clean = np.argsort(D_clean, axis=1)[:, :k]
    nn_noisy = np.argsort(D_noisy, axis=1)[:, :k]

    overlaps = []
    for a, b in zip(nn_clean, nn_noisy):
        overlap = len(set(a).intersection(set(b))) / k
        overlaps.append(overlap)

    overlaps = np.array(overlaps)

    print(f"Average top-{k} nearest-neighbor overlap = {np.mean(overlaps):.4f}")
    print(f"Median top-{k} nearest-neighbor overlap = {np.median(overlaps):.4f}")

    return overlaps


def noise_estimation(
    dataset,
    embedding_pkl,
    noise_ratio=0.1,
    seed=42,
    pairwise_median=None,
    sample_pairs=50000,
    renormalize=False,
):
    target_name = embedding_pkl[:-4]+f"_{int(noise_ratio * 100)}pct_Seed_{seed}.pkl"
    pkl_path = os.path.join(f"E:/Project/CurrentDataset/result/embedding/{dataset}/", embedding_pkl)
    target_path = os.path.join(f"E:/Project/CurrentDataset/result/embedding/{dataset}/", target_name)
    with open(pkl_path, "rb") as f:
        embeddings = pickle.load(f)
    Z = []
    for item in embeddings:
        Z.append(item[1][0])
    Z = np.array(Z, dtype=np.float64)
    noisy, sigma, m = add_gaussian_noise_to_table_embeddings(
        Z,
        noise_ratio,
        pairwise_median=pairwise_median,
        sample_pairs=sample_pairs,
        seed=seed,
        renormalize=renormalize
    )
    new_embeddings = [(item[0], np.array([noisy[index]])) for index, item in enumerate(embeddings)]

    # sanity check
    """stats = pairwise_distance_sanity_check(
        Z,
        noisy,
        sample_pairs=50000,
        seed=123
    )"""

    perturbation_scale_check(
        Z,
        noisy,
        noise_ratio=noise_ratio,
        m=m,
        sample_pairs=50000,
        seed=123
    )
    pairwise_rank_correlation_check(Z,
        noisy,
        sample_pairs=50000,
        seed=42)
    nearest_neighbor_overlap_check(Z,
        noisy, k=10)


    with open(target_path, "wb") as f:
        pickle.dump(new_embeddings,f)

data = "WDC"
noise_list =[0]#0.02,0.04,0.06,0.08,0.1, 0.2, 0.3, 0.4, 0.5
seeds = [43,45,44]
for noise in noise_list:
    for seed in seeds:
        print(f"current noise level = {noise}")
        new_name = "Pretrain_sbert_head_column_none_weighted_False.pkl"
        noise_estimation(
            data,
            new_name,
            noise_ratio=noise,
            seed=seed,
            pairwise_median=None,
            sample_pairs=50000,
            renormalize=False,
        )
