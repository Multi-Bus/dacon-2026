# DACON 2026 — 백본 교체 실험 (팀 분업 가이드)

현재 로컬 스택 OOF **0.7378** (klue-base + m1v2 + m2v3 + m2cat + s1). 다음 레버는
"klue-base보다 강한 transformer 백본을 찾는 것"이다. GPU가 1대뿐이면 백본 후보들을
순차로 돌려야 해서 총 12시간 가까이 걸리지만, 인원을 나누면 3시간대로 줄일 수 있다.
그래서 이 실험만 떼어내 분업한다.

- **나 (로컬 4060 Ti)**: klue-large fold0 게이트 — 이미 백그라운드 실행 중, **건드리지 않음**.
- **GPU 팀원 A**: mDeBERTa-v3-base fold0
- **GPU 팀원 B**: KR-ELECTRA → electra-kor-base fold0 (순차, 같은 GPU에서 2개)
- **CPU 팀원 C**: 기존 5개 뷰로 스택 조합 재탐색 (forward-selection + isotonic 보정) — GPU 불필요

각자 **fold0(전체 5-fold 중 1개)만** 학습해서 klue-base의 fold0 점수(0.709)와 비교한다.
전체 5-fold를 다 돌리는 게 아니므로 실험당 2~5시간 안에 끝난다. 이긴 후보만 나중에
5-fold 전체를 마저 학습해 스택에 편입한다.

---

## 0. 공통 준비 (전원)

```bash
git clone https://github.com/Multi-Bus/dacon-2026.git
cd dacon-2026
```

레포에는 **코드 + 학습 데이터(train.jsonl 등) + 기존 OOF 결과**만 들어있다.
(`submit/model`처럼 이미 배포된 무거운 모델 파일, 과거 제출 zip은 각자 로컬에만 있고
레포에는 올리지 않았다 — 지금 실험엔 필요 없음.)

레포 루트 기준 폴더 구조:
```
dacon-2026/
├── dacon_data/data/     # train.jsonl(7만 건), train_labels.csv, test.jsonl
├── train/
│   ├── features.py      # 텍스트/메타 피처 빌더 (건드릴 필요 없음)
│   ├── cv.py             # GroupKFold(5) — 전원 동일 분할 보장의 핵심
│   ├── bakeoff_fold0.py  # GPU 팀원이 실행할 스크립트
│   ├── stack_v2.py       # CPU 팀원이 실행할 스크립트
│   └── oof/              # 기존 5개 뷰의 OOF 확률 (스택 재탐색용)
├── requirements_gpu.txt
└── requirements_cpu.txt
```

**중요**: `cv.py`의 `make_group_folds(ids, y, n_splits=5, seed=42)`가 전원 동일한 fold0을
만들어준다. 이 파일을 수정하거나 다른 seed를 쓰면 비교가 무효가 되니 그대로 쓸 것.

---

## 1. GPU 팀원 A — mDeBERTa-v3-base

```bash
pip install -r requirements_gpu.txt
```

**주의**: mDeBERTa 토크나이저는 `protobuf`+`sentencepiece`가 없으면
`Error parsing line ... spm.model` 에러가 난다 (나도 겪었음). 위 requirements에 이미 포함.

### 1) 스모크 테스트 먼저 (5분, VRAM/속도 확인)
```bash
cd train
python bakeoff_fold0.py --model microsoft/mdeberta-v3-base --tag mdeberta --bs 8 --grad_ckpt --smoke
```
60스텝만 돌고 종료하며 `peak VRAM`과 `예상 X분/epoch`를 출력한다.

- **VRAM 8GB급 GPU**: `--bs 8 --grad_ckpt` 유지 (내 4060Ti 기준 mDeBERTa 278M이면 OOM 안 남)
- **VRAM 12GB 이상**: `--grad_ckpt` 빼고 `--bs 16`으로 — 더 빠름
- OOM 나면: `--bs 4 --accum 2` (유효 배치 8 유지)

### 2) 본 실행 (예상 4~5시간, 5 epoch)
```bash
python bakeoff_fold0.py --model microsoft/mdeberta-v3-base --tag mdeberta --bs 8 --grad_ckpt
```
백그라운드 권장:
```bash
nohup python bakeoff_fold0.py --model microsoft/mdeberta-v3-base --tag mdeberta --bs 8 --grad_ckpt > mdeberta.log 2>&1 &
```

### 3) 결과 확인
```bash
tail -5 mdeberta.log
```
`[mdeberta] fold0 F1=0.XXXX (klue-base fold0=0.709)` 형태로 나온다.
**결과 파일**: `train/oof/mdeberta_fold0.npz` (확률 행렬, 약 2.5MB) — 이게 회수 대상.

---

## 2. GPU 팀원 B — KR-ELECTRA → electra-kor-base (같은 GPU, 순차)

```bash
pip install -r requirements_gpu.txt
cd train
```

이 둘은 110M급(klue-base와 비슷한 크기)이라 grad_ckpt 없이도 웬만한 GPU에서 돌아간다.

### 1) 스모크
```bash
python bakeoff_fold0.py --model snunlp/KR-ELECTRA-discriminator --tag kr_electra --bs 16 --smoke
```
OOM 나면 `--bs 8` 또는 `--bs 8 --accum 2`로.

### 2) 본 실행 — 첫 번째 모델 (예상 1.5~2.5시간)
```bash
nohup python bakeoff_fold0.py --model snunlp/KR-ELECTRA-discriminator --tag kr_electra --bs 16 > kr_electra.log 2>&1 &
```
**이 프로세스가 끝난 뒤에** 다음 모델 시작 (같은 GPU에서 동시 실행하면 OOM):
```bash
nohup python bakeoff_fold0.py --model kykim/electra-kor-base --tag kor_electra --bs 16 > kor_electra.log 2>&1 &
```

### 3) 결과 확인
```bash
tail -5 kr_electra.log kor_electra.log
```
**결과 파일 2개**: `train/oof/kr_electra_fold0.npz`, `train/oof/kor_electra_fold0.npz`

---

## 3. CPU 팀원 C — 스택 조합 재탐색 (GPU 불필요)

```bash
pip install -r requirements_cpu.txt
cd train
python stack_v2.py
```

**5~10분 안에 끝난다.** 하는 일:
- 기존 5개 뷰(klue-base, m1v2, m2v3, m2cat, s1)의 OOF를 가지고
  forward-selection으로 "어떤 조합이 스택 점수를 최대화하는지" 자동 탐색
- 각 뷰에 per-class isotonic 보정을 적용했을 때 더 나아지는지 A/B 비교

출력 예시:
```
=== A. 원본 확률 forward-selection ===
  + klue   → ['klue']  스택OOF=0.7145
  + m2v3   → ['klue', 'm2v3']  스택OOF=0.73xx
  ...
[RAW] 선택 조합=[...]  OOF=0.73xx

=== B. isotonic 보정 후 forward-selection ===
  ...
```

**출력 전체를 캡처해서 공유**(터미널 로그 복붙 또는 `python stack_v2.py > stack_v2_result.log` 후 파일 전달).
이 결과로 "현재 5뷰가 최적 조합인지, 보정이 도움되는지"를 확정한다.

---

## 4. 결과 회수 방법

각자 작업이 끝나면 **자기 브랜치로 커밋 후 push** (충돌 방지):

```bash
git checkout -b bakeoff-<자기이름>      # 예: bakeoff-teamB
git add train/oof/<자기_tag>_fold0.npz   # 예: train/oof/mdeberta_fold0.npz
git commit -m "add <tag> fold0 OOF"
git push origin bakeoff-<자기이름>
```

CPU 팀원은:
```bash
git checkout -b stack-v2-result
git add stack_v2_result.log            # (직접 만든 로그 파일)
git commit -m "stack_v2 forward-selection result"
git push origin stack-v2-result
```

파일이 각각 2~5MB로 작아서 굳이 클라우드 전송 없이 **깃허브 push만으로 충분**하다.
push 후 아무 채널로든 "브랜치 올렸다"고만 알려주면 내가 받아서 로컬 스택에 합친다.

---

## 5. 하지 말아야 할 것

- `cv.py`, `features.py` **수정 금지** — 전원 동일 fold 분할이 비교의 전제.
- `main` 브랜치에 직접 push 금지 — 위처럼 각자 브랜치 사용.
- `dacon_data/data/` 안의 파일 수정 금지 (원본 유지).
- 큰 파일(모델 체크포인트, zip 등) 실수로 커밋하지 않기 — `.gitignore`가 기본 차단하지만,
  `git add -A`처럼 전체 add는 하지 말고 위에 적힌 파일만 개별로 `git add`할 것.

---

## 6. 왜 이 순서인가 (배경)

- 지금 스택 천장은 클래스 5개 조합 다 짜내도 **0.7378**. 트리/선형 뷰(m1/m2/m2cat/s1)를
  전부 빼면 스택이 0.65로 떨어지고, transformer 하나(klue) 단독이 0.7145로 그 4개를 합친 것보다
  강함 — 실측 확인됨. 즉 **성능을 가르는 건 백본**이고, 나머지 뷰는 이미 짜낼 만큼 짜낸 상태.
- 그래서 klue-base보다 강한 백본을 찾는 게 최대 레버. mDeBERTa(다국어 250k vocab, DeBERTa-v3
  구조)가 1순위 후보, KR-ELECTRA/electra-kor는 klue와 다른 사전학습이라 다양성 후보.
- fold0만 보는 이유: 5-fold 전부 돌리면 후보당 10시간대라 바로 배제 안 되는 후보에 시간을 낭비함.
  fold0 하나로 "가망 있는지"만 빠르게 걸러내고, 이긴 후보만 5-fold 마저 돌린다.
