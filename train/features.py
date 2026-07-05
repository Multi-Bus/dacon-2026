"""피처 빌더 — 학습/추론 공유 정본(single source of truth).

이 모듈의 build_text / meta_dict / META_CAT_COLS / META_NUM_COLS / ACTIONS 는
제출용 submit/script.py 에 **그대로 인라인 복사**된다(baseline의 자기완결 구조 유지).
두 곳의 로직이 항상 동일해야 train-serve skew 가 없다.

- build_text(sample): current_prompt + 최근 history K턴 → 역할 마커 직렬화 (텍스트 뷰)
- meta_dict(sample): session_meta + action 시퀀스 → 구조화 피처 dict (구조화 뷰)
"""
from __future__ import annotations

import json
import csv

# 예측 대상 14개 클래스
ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

# language_mix / open_files 확장자 분해에 쓸 주요 확장자
EXTS = ["py", "ts", "tsx", "js", "jsx", "yaml", "yml", "json",
        "md", "sh", "tf", "dockerfile", "css", "go", "rs", "java"]

# 구조화 뷰 컬럼 정의 (M1 인코딩·M2 입력에서 참조)
META_CAT_COLS = ["user_tier", "language_pref", "last_ci", "top_lang",
                 "last_action", "last2_action"]
META_NUM_COLS = (
    ["loc", "budget", "turn_index", "elapsed", "git_dirty", "n_open",
     "top_ratio", "hist_len", "n_user_turns", "n_action_turns",
     "budget_per_turn", "elapsed_per_turn", "ci_failed_and_dirty"]
    + ["mix_" + e for e in EXTS]
    + ["openext_" + e for e in EXTS]
    + ["act_cnt_" + a for a in ACTIONS]   # history 내 action 카운트 14차원
)


# ---------- 데이터 로드 ----------
def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_labels(path):
    return {r["id"]: r["action"]
            for r in csv.DictReader(open(path, encoding="utf-8"))}


def session_group(sample_id: str) -> str:
    """세션 base id (GroupKFold 그룹 키). 'sess_..._<n>-step_08' -> 'sess_..._<n>'."""
    return str(sample_id).split("-step_")[0]


# ---------- 텍스트 뷰 ----------
def _history_split(history):
    users, actions = [], []
    for t in history or []:
        if t.get("role") == "user":
            users.append(t)
        elif t.get("role") == "assistant_action":
            actions.append(t)
    return users, actions


def build_text(sample, k: int = 4) -> str:
    """현재 발화 + 최근 history K턴을 역할 마커로 직렬화."""
    parts = ["[CUR] " + str(sample.get("current_prompt", "") or "")]
    users, actions = _history_split(sample.get("history", []))
    for t in users[-k:]:
        parts.append("[U] " + str(t.get("content", "") or ""))
    for t in actions[-k:]:
        parts.append("[ACT] " + str(t.get("name", "") or "")
                     + " " + str(t.get("result_summary", "") or ""))
    return " ".join(parts)


# ---------- 구조화 뷰 (session_meta 집중) ----------
def _last_actions(history):
    names = [t.get("name", "none") for t in (history or [])
             if t.get("role") == "assistant_action"]
    last = names[-1] if names else "none"
    last2 = names[-2] if len(names) >= 2 else "none"
    return last, last2, names


def meta_dict(sample) -> dict:
    """session_meta + action 시퀀스 → 구조화 피처 dict."""
    sm = sample.get("session_meta", {}) or {}
    ws = sm.get("workspace", {}) or {}
    lm = ws.get("language_mix", {}) or {}
    of = ws.get("open_files", []) or []
    history = sample.get("history", []) or []
    users, actions = _history_split(history)
    last, last2, act_names = _last_actions(history)

    turn_index = sm.get("turn_index", 0) or 0
    budget = sm.get("budget_tokens_remaining", 0) or 0
    elapsed = sm.get("elapsed_session_sec", 0) or 0
    git_dirty = int(bool(ws.get("git_dirty", False)))
    ci = ws.get("last_ci_status", "none")
    denom = turn_index if turn_index else 1

    row = {
        # 범주형
        "user_tier": sm.get("user_tier", "?"),
        "language_pref": sm.get("language_pref", "?"),
        "last_ci": ci,
        "top_lang": (max(lm, key=lm.get) if lm else "?"),
        "last_action": last,
        "last2_action": last2,
        # 수치 (session_meta 핵심)
        "loc": ws.get("loc", 0) or 0,
        "budget": budget,
        "turn_index": turn_index,
        "elapsed": elapsed,
        "git_dirty": git_dirty,
        "n_open": len(of),
        "top_ratio": (max(lm.values()) if lm else 0.0),
        "hist_len": len(history),
        "n_user_turns": len(users),
        "n_action_turns": len(actions),
        # 파생 상호작용
        "budget_per_turn": budget / denom,
        "elapsed_per_turn": elapsed / denom,
        "ci_failed_and_dirty": int(ci == "failed") * git_dirty,
    }
    # language_mix 비율 분해
    for e in EXTS:
        row["mix_" + e] = float(lm.get(e, 0.0))
    # open_files 확장자 존재 플래그
    of_low = [str(f).lower() for f in of]
    for e in EXTS:
        row["openext_" + e] = int(any(f.endswith("." + e) for f in of_low))
    # history action 카운트 14차원
    from collections import Counter
    cnt = Counter(act_names)
    for a in ACTIONS:
        row["act_cnt_" + a] = int(cnt.get(a, 0))
    return row


# ---------- 강화 구조화 뷰 (M2 전용, 탐색 액션 구분 신호 집중) ----------
# 탐색/행동 의도 키워드 (current_prompt 이산 신호). 한/영 혼합.
_KW = {
    "read": ["read", "show", "open", "display", "print", "look", "see", "view",
             "content", "보여", "읽어", "출력", "확인", "impl"],
    "search": ["find", "search", "grep", "where", "locate", "which", "찾", "검색", "어디"],
    "list": ["list", " ls", "directory", "folder", "tree", "목록", "디렉", "폴더"],
    "glob": ["glob", "pattern", "wildcard", "*.", "all files", "패턴", "매칭"],
    "edit": ["edit", "change", "update", "modify", "fix", "refactor", "rename",
             "수정", "고쳐", "바꿔", "변경", "리팩"],
    "write": ["write", "create", "add ", "new file", "generate", "작성", "만들", "생성", "추가"],
    "test": ["test", "pytest", "unittest", "테스트"],
    "run": ["run", "execute", "build", "compile", "npm", "make", "실행", "빌드", "돌려"],
    "lint": ["lint", "typecheck", "mypy", "format", "type check", "린트", "타입"],
    "ask": ["should i", "do you want", "?", "물어", "질문", "어떻게 할"],
    "plan": ["plan", "steps", "approach", "design", "계획", "설계", "단계"],
    "web": ["google", "documentation", "docs", "latest", "online", "changelog", "문서", "최신"],
}
_KW_KEYS = list(_KW.keys())

# 강화 뷰 컬럼 (기존 meta_dict + 아래 추가분)
RICH_CAT_COLS = META_CAT_COLS + ["last3_action", "last_path_ext"]
RICH_NUM_COLS = META_NUM_COLS + (
    ["cp_len", "cp_wc", "cp_has_code", "cp_has_path", "cp_has_q", "cp_has_num",
     "last_eq_last2", "n_uniq_act", "last_has_path", "last_user_len",
     "act_switch_rate"]
    + ["kw_" + k for k in _KW_KEYS]
)


def _path_ext(path):
    p = str(path or "").lower().rsplit(".", 1)
    return p[1] if len(p) == 2 and 1 <= len(p[1]) <= 5 else "none"


def meta_rich(sample) -> dict:
    """meta_dict 확장 — M2(구조화 전문가)용. 탐색 액션 구분 이산 신호 추가."""
    row = meta_dict(sample)
    history = sample.get("history", []) or []
    _, _, act_names = _last_actions(history)
    cp = str(sample.get("current_prompt", "") or "").lower()

    # action 시퀀스 심화
    row["last3_action"] = act_names[-3] if len(act_names) >= 3 else "none"
    row["last_eq_last2"] = int(len(act_names) >= 2 and act_names[-1] == act_names[-2])
    row["n_uniq_act"] = len(set(act_names))
    row["act_switch_rate"] = (
        sum(1 for a, b in zip(act_names, act_names[1:]) if a != b) / (len(act_names) - 1)
        if len(act_names) >= 2 else 0.0)

    # 직전 action의 args.path 확장자 (탐색 대상 힌트)
    last_act = None
    for t in reversed(history):
        if t.get("role") == "assistant_action":
            last_act = t
            break
    last_path = (last_act or {}).get("args", {}).get("path") if last_act else None
    row["last_path_ext"] = _path_ext(last_path)
    row["last_has_path"] = int(bool(last_path))

    # 직전 user 발화 길이
    last_user = ""
    for t in reversed(history):
        if t.get("role") == "user":
            last_user = str(t.get("content", "") or "")
            break
    row["last_user_len"] = len(last_user)

    # current_prompt 텍스트 통계
    row["cp_len"] = len(cp)
    row["cp_wc"] = len(cp.split())
    row["cp_has_code"] = int("```" in cp or "    " in cp)
    row["cp_has_path"] = int("/" in cp or any("." + e in cp for e in EXTS))
    row["cp_has_q"] = int("?" in cp)
    row["cp_has_num"] = int(any(ch.isdigit() for ch in cp))

    # 의도 키워드 플래그
    for k in _KW_KEYS:
        row["kw_" + k] = int(any(w in cp for w in _KW[k]))
    return row


import re as _re
_RE_WILD = _re.compile(r"[*?]|\*\.\w+|\*_|_\*|\[[^\]]+\]")
_RE_FILE = _re.compile(r"\b[\w./-]+\.(py|ts|tsx|js|jsx|go|rs|java|yaml|yml|json|md|sh|tf|toml|gradle|txt|cfg|ini|xml|html|css)\b")
_RE_QUOTE = _re.compile(r"[\"'`][^\"'`]{2,}[\"'`]")
_DIRWORDS = ["directory", "folder", " dir ", "tree", " ls", "루트", "root", "디렉", "폴더", "구조", "뭐가 있", "뭐 있", "어떤 파일", "파일들", "목록"]
_FINDALL = ["find all", "every", "모든", "전부", "다 찾", "죄다", "싹 다", "matching", "형식으로", "패턴"]
_WHERE = ["where", "어디", "어느", "위치", "locate", "찾아"]

# 탐색 4클래스 disambiguation 전용 sharp 피처 (M2 v3용)
SHARP_NUM_COLS = ["sh_wild", "sh_nfile", "sh_quote", "sh_dirword", "sh_findall",
                  "sh_where", "sh_grep_kw", "sh_read_kw", "sh_ext_in_prompt"]
RICH2_NUM_COLS = RICH_NUM_COLS + SHARP_NUM_COLS
RICH2_CAT_COLS = RICH_CAT_COLS


def meta_rich2(sample) -> dict:
    """meta_rich + 탐색 disambiguation sharp 피처."""
    row = meta_rich(sample)
    cp = str(sample.get("current_prompt", "") or "")
    cpl = cp.lower()
    files = _RE_FILE.findall(cp)
    row["sh_wild"] = int(bool(_RE_WILD.search(cp)))                      # 와일드카드 → glob
    row["sh_nfile"] = len(files)                                        # 파일명 언급 수 → read
    row["sh_quote"] = int(bool(_RE_QUOTE.search(cp)))                   # 인용문 → grep
    row["sh_dirword"] = int(any(w in cpl for w in _DIRWORDS))          # 디렉토리 단어 → list
    row["sh_findall"] = int(any(w in cpl for w in _FINDALL))           # 광범위 검색 → glob/grep
    row["sh_where"] = int(any(w in cpl for w in _WHERE))               # 위치 질의 → grep
    row["sh_grep_kw"] = int("grep" in cpl or "search" in cpl or "검색" in cpl)
    row["sh_read_kw"] = int(any(w in cpl for w in ["read", "show", "open", "cat", "보여", "읽", "열어"]))
    row["sh_ext_in_prompt"] = int(bool(files))
    return row
