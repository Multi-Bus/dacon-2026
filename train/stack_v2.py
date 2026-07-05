"""Phase B v2 — 확장 뷰 풀 forward-selection + isotonic 보정 A/B.

기존(klue,m1v2,m2v3)에 신규 뷰(s1,m2cat) 추가 → greedy 선택으로 최적 조합 재결정.
보정: 각 뷰 확률을 per-class isotonic 으로 nested-OOF 보정 후 재스택 → 이득 있는지 실측.
klue-large OOF 도착 시 CANDIDATES 에 추가하고 재실행하면 됨.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
from cv import make_group_folds
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

OOF_DIR = os.path.join(HERE, "oof")

CANDIDATES = {
    "klue": "m3_klue_oof.npy",
    "m1v2": "m1v2_oof.npy",
    "m2v3": "m2v3_oof.npy",
    "m2cat": "m2cat_oof.npy",
    "s1": "s1_oof.npy",
    # klue-large 도착 시: "klueL": "m3_klueL_oof.npy",
}


def mf(y, p):
    return f1_score(y, np.array(F.ACTIONS)[p.argmax(1)], labels=F.ACTIONS,
                    average="macro", zero_division=0)


def load_views():
    import json
    y = np.load(os.path.join(OOF_DIR, "oof_y.npy"), allow_pickle=True)
    ids = json.load(open(os.path.join(OOF_DIR, "oof_ids.json")))
    views = {}
    for name, fn in CANDIDATES.items():
        p = os.path.join(OOF_DIR, fn)
        if os.path.exists(p):
            arr = np.load(p)
            if arr.shape[0] == len(y) and not np.allclose(arr.sum(), 0):
                views[name] = arr
                print(f"  {name:6s} 단독 OOF={mf(y, arr):.4f}")
    return y, ids, views


def calibrate_oof(view, y, folds):
    """per-class isotonic, nested-OOF (누수 없이 보정된 확률 반환)."""
    n, n_cls = view.shape
    out = np.zeros_like(view)
    for tr, va in folds:
        for c in range(n_cls):
            iso = IsotonicRegression(out_of_bounds="clip")
            yc = (y[tr] == F.ACTIONS[c]).astype(float)
            iso.fit(view[tr, c], yc)
            out[va, c] = iso.transform(view[va, c])
    # 행 정규화
    s = out.sum(1, keepdims=True); s[s == 0] = 1
    return out / s


def stack(y, folds, mats):
    ci = {c: i for i, c in enumerate(F.ACTIONS)}
    X = np.hstack(mats); o = np.zeros((len(y), len(F.ACTIONS)), np.float32)
    for tr, va in folds:
        c = LogisticRegression(max_iter=1000, C=1.0); c.fit(X[tr], y[tr])
        for j, cc in enumerate(c.classes_):
            o[va, ci[cc]] = c.predict_proba(X[va])[:, j]
    return mf(y, o)


def forward_select(y, folds, views, label):
    names = list(views.keys())
    selected, best = [], -1
    remaining = set(names)
    while remaining:
        cand = (None, best)
        for nm in remaining:
            trial = selected + [nm]
            f = stack(y, folds, [views[t] for t in trial]) if len(trial) >= 2 else mf(y, views[nm])
            if f > cand[1]:
                cand = (nm, f)
        if cand[0] is None:
            break
        selected.append(cand[0]); remaining.discard(cand[0]); best = cand[1]
        print(f"  + {cand[0]:6s} → {selected}  스택OOF={best:.4f}", flush=True)
    print(f"[{label}] 선택 조합={selected}  OOF={best:.4f}\n", flush=True)
    return selected, best


def main():
    print("[stack_v2] 뷰 로드:")
    y, ids, views = load_views()
    folds = make_group_folds(ids, y, n_splits=5)

    print("\n=== A. 원본 확률 forward-selection ===")
    forward_select(y, folds, views, "RAW")

    print("=== B. isotonic 보정 후 forward-selection ===")
    cal = {nm: calibrate_oof(v, y, folds) for nm, v in views.items()}
    for nm in views:
        print(f"  {nm:6s} 보정후 단독={mf(y, cal[nm]):.4f}")
    forward_select(y, folds, cal, "CALIB")


if __name__ == "__main__":
    main()
