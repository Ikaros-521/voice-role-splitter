import numpy as np
import pandas as pd
import pytest

from vrs import cluster as CL
from vrs.config import Config


class Cfg:
    """minimal stand-in for the parts cluster.py reads"""
    seed = 0
    ks = [2, 3, 4]
    score = "cosine"
    linkage = "average"
    min_cluster = 3
    center = True
    cluster_min_speech = 0.5
    silver_pct = 0.5
    window, hop = 3.0, 1.5


def blobs(seed=0, per=40, dim=16, sep=1.0):
    """three clearly separated groups -> clustering should find them"""
    rng = np.random.default_rng(seed)
    out, lab = [], []
    for k in range(3):
        cent = np.zeros(dim, np.float32)
        cent[k] = sep
        out.append(cent + rng.normal(0, 0.05, (per, dim)).astype(np.float32))
        lab += [k] * per
    return np.concatenate(out, 0), np.array(lab)


def test_l2_and_center_shapes():
    F = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    F = F + 3.0                                   # a large shared offset dominates raw cosine
    assert np.allclose(np.linalg.norm(CL.l2(F), axis=1), 1, atol=1e-5)
    raw = CL.l2(F)
    C = CL.center(F)
    off = lambda M: float((M @ M.T)[~np.eye(len(M), dtype=bool)].mean())
    assert off(C) < off(raw), "centering must remove the common direction that inflates cosine"


def test_asnorm_is_symmetric():
    E = CL.l2(np.random.default_rng(1).normal(size=(30, 8)).astype(np.float32))
    S = E @ E.T
    N = CL.asnorm(S, topk=5)
    assert np.allclose(N, N.T, atol=1e-5)


def test_merge_small_folds_tiny_clusters_and_terminates():
    """regression: an emptied label used to stay below the floor forever and hang the loop"""
    E = np.eye(6, dtype=np.float32)
    labels = np.array([0, 0, 0, 0, 1, 2])
    out = CL.merge_small(labels, E, floor=3)
    assert len(np.unique(out)) < 3
    assert set(out) <= {0, 1, 2}
    assert (np.bincount(out)[np.bincount(out) > 0] >= 3).all()


def test_merge_small_with_all_tiny_converges():
    E = np.eye(5, dtype=np.float32)
    out = CL.merge_small(np.arange(5), E, floor=3)
    assert len(np.unique(out)) == 1


def test_centroid_map_and_nearest_labels_roundtrip():
    E = CL.l2(np.array([[1, 0], [0.98, 0.02], [0, 1]], np.float32))
    cent = CL.centroid_map(np.array([0, 0, 1]), E)
    assert set(cent) == {0, 1}
    assert list(CL.nearest_labels(E, cent)) == [0, 0, 1]


def test_fuse_weights_sum_to_one():
    cfg = Cfg()
    embs = {"a": np.eye(4, 3).astype(np.float32), "b": np.eye(4, 3).astype(np.float32)}
    A, per, w = CL.fuse(embs, cfg)
    assert abs(sum(w.values()) - 1) < 1e-6
    assert A.shape == (4, 3) and np.allclose(np.linalg.norm(A, axis=1), 1, atol=1e-5)


def test_sweep_and_choose_find_three_groups(tmp_path):
    X, ytrue = blobs(per=30)
    cfg = Cfg()
    cfg.ks = [2, 3, 4, 6]
    embs = {"a": X, "b": X}
    A, per, _ = CL.fuse(embs, cfg)
    has = np.ones(len(A), bool)
    Z, sweep, E = CL.sweep(A, per, has, cfg, verbose=False)
    assert {"k", "n_clusters", "agree_kappa", "silhouette"} <= set(sweep.columns)
    k, best = CL.choose(sweep)
    assert k in (2, 3, 4)
    from scipy.cluster.hierarchy import fcluster
    lab = CL.merge_small(fcluster(Z, t=3, criterion="maxclust") - 1, E, 5)
    assert len(np.unique(lab)) == 3
    # each true group lands in one cluster
    assert all(len(pd.Series(lab[ytrue == g]).unique()) == 1 for g in (0, 1, 2))


def test_choose_honours_force_k(tmp_path):
    X, _ = blobs(per=20)
    cfg = Cfg()
    cfg.ks = [2, 3, 5]
    A, per, _ = CL.fuse({"a": X}, cfg)
    Z, sweep, E = CL.sweep(A, per, np.ones(len(A), bool), cfg, verbose=False)
    k, _ = CL.choose(sweep, force_k=5)
    assert k == 5
    with pytest.raises(ValueError, match="not in the sweep"):
        CL.choose(sweep, force_k=99)
