import numpy as np
import pytest
import pandas as pd

from vrs import scoring as SC


class Cfg:
    k_neighbors = 2
    silver_pct = 0.5
    keep_margin = 0.05
    assign_margin = 0.05
    only_unassigned = True


def labels(n_per=8):
    """two roles, each split over two clusters, in an orthonormal-ish space"""
    rows, cl, role, pct = [], [], [], []
    basis = np.eye(6, dtype=np.float32)
    for r, (a, b) in enumerate([(0, 1), (2, 3)]):          # role A uses dims 0/1, role B 2/3
        for c, d in enumerate((a, b)):                     # two clusters per role
            for i in range(n_per):
                v = basis[d] + 0.05 * basis[4] * (i / n_per)
                rows.append(v)
                cl.append(r * 2 + c)
                role.append(f"role{r}")
                pct.append(0.6 + 0.4 * (i % 3) / 3)
    E = np.stack(rows)
    df = pd.DataFrame(dict(file=[f"f{i:03d}.wav" for i in range(len(E))], cluster=cl,
                           role=role, own_pct=pct, speech_sec=[2.0] * len(E),
                           embedded=[True] * len(E), has_vector=[True] * len(E),
                           centroid_score=[0.7] * len(E)))
    return E, df


def names(df):
    return {int(c): r for c, r in zip(df.cluster, df.role)}


def test_topk_mean_averages_largest():
    M = np.array([[0.5, 0.1, 0.9, 0.2]], np.float32)
    assert SC.topk_mean(M, 2)[0] == pytest.approx(float(np.mean([0.9, 0.5])), abs=1e-6)


def test_score_never_cites_self_and_cross_leaves_own_cluster_out():
    E, df = labels()
    cfg = Cfg()
    ref, ref_role, multi, _ = SC.build_references(df, names(df), cfg)
    rn, Sl, Sx = SC.score(E, ref, ref_role, df, cfg)
    j = {r: i for i, r in enumerate(rn)}
    own = np.array([Sl[i, j[df.role.iloc[i]]] for i in range(len(df))])
    # an unmasked self-similarity would be exactly 1.0 for unit vectors
    assert not np.isclose(own, 1.0, atol=1e-6).any(), "a file was allowed to reference itself"
    # cross score drops the whole own cluster, so it can never exceed the cohesive score
    for i in range(len(df)):
        assert Sx[i, j[df.role.iloc[i]]] <= Sl[i, j[df.role.iloc[i]]] + 1e-6
    assert multi == ["role0", "role1"], "roles spanning >=2 clusters must be cross-testable"


def test_decide_only_unassigned_leaves_clustered_files_alone():
    E, df = labels()
    cfg = Cfg()
    ref, ref_role, multi, _ = SC.build_references(df, names(df), cfg)
    rn, Sl, Sx = SC.score(E, ref, ref_role, df, cfg)
    d = SC.decide(df, cfg, rn, Sl, Sx, multi)
    assert set(d[df.cluster >= 0].verdict) == {"keep"}


def test_decide_assigns_unresolved_files_when_margin_is_clear():
    E, df = labels()
    cfg = Cfg()
    ref, ref_role, multi, _ = SC.build_references(df, names(df), cfg)
    rn, Sl, Sx = SC.score(E, ref, ref_role, df, cfg)
    df.loc[df.index[:4], ["cluster", "role"]] = [-1, "unresolved"]   # no cluster -> score-only
    df.loc[df.index[:4], "own_pct"] = np.nan
    d = SC.decide(df, cfg, rn, Sl, Sx, multi)
    head = d.iloc[:4]
    assert set(head.verdict) <= {"assign", "manual"}
    assert (head[head.verdict == "assign"].suggested.isin(["role0", "role1"])).all()


def test_apply_never_overwrites_with_a_mere_suggestion():
    E, df = labels()
    cfg = Cfg()
    cfg.only_unassigned = False
    ref, ref_role, multi, _ = SC.build_references(df, names(df), cfg)
    rn, Sl, Sx = SC.score(E, ref, ref_role, df, cfg)
    d = SC.decide(df, cfg, rn, Sl, Sx, multi)
    d.loc[:, "verdict"] = "propose"
    d.loc[:, "suggested"] = "someone_else"
    final, note = SC.apply(d, df)
    assert list(final) == list(df.role), "propose must not silently reassign"
    assert all(n.startswith("建议改判") for n in note)


def test_calibrate_reports_accuracy_per_margin_bucket():
    E, df = labels()
    cfg = Cfg()
    ref, ref_role, multi, _ = SC.build_references(df, names(df), cfg)
    rn, Sl, Sx = SC.score(E, ref, ref_role, df, cfg)
    gold = pd.DataFrame(dict(file=df.file[:8], role=df.role[:8]))
    cal, curve = SC.calibrate(gold, df, rn, Sl, Sx, set(multi))
    assert {"hit_cross", "cross_margin", "testable"} <= set(cal.columns)
    assert 0 <= cal.hit_cross.mean() <= 1
    assert curve.n.sum() == int(cal.testable.sum())
