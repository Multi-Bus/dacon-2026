"""검증 유틸 — 세션 단위 GroupKFold.

같은 세션(id의 '-step_' 앞부분)이 여러 스텝으로 반복되므로, 일반 KFold는
인접 스텝 간 history 중복으로 누수가 생긴다. 세션을 그룹으로 묶어 분할한다.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from features import session_group


def make_group_folds(ids, y, n_splits=5, seed=42):
    """(train_idx, val_idx) 리스트 반환. 세션이 fold를 넘나들지 않음."""
    groups = np.array([session_group(i) for i in ids])
    y = np.asarray(y)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))
