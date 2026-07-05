# DACON 2026 — 백본 교체 실험 (팀 분업 가이드) · v2

> **⚠️ 이 문서는 재실행용으로 갱신됨(2026-07-06).** 이전에 돌린 DeBERTa 결과는 무효(NaN 버그)이고,
> 환경을 평가 서버에 맞춰 통일했다. **처음부터 다시 clone → 아래 순서대로 재실행**할 것.

## 0. 목표와 원칙 (바뀐 부분)

- **목표: 예선 리더보드 Macro-F1 0.8 돌파.** (현재 상위권 대부분 0.79, 기간 내 돌파 예상 → 0.8을 뚫어야 입상권.)
- **이 대회는 예선(리더보드 점수)만으로 입상이 갈린다. 추론 속도는 신경 쓰지 않는다.**
  → 무거운 모델이라도 성능(Macro-F1)만 높으면 됨. 크기·속도 제약은 zip 1GB / 추론 10분 한도만 지키면 OK.
- **lookup(세션 누수) 전략은 폐기.** 정정당당한 모델 성능으로만 승부.

현재 로컬 스택 OOF **0.7378**(klue-base+m1v2+m2v3+m2cat+s1). 다음 레버 = **klue보다 강한 백본**.
klue 단독 0.7145가 트리/선형 4개 합(0.6512)보다 강함이 실측됨 → 강한 transformer 하나가 성능을 좌우.
그래서 백본 후보들을 fold0만 먼저 돌려 klue(fold0 0.7143)와 비교하고, 이긴 것만 5-fold 완주한다.

## 0-1. 🚨 환경: 평가 서버와 동일하게 통일 (반드시 준수)

이전에 쓰던 transformers 5.12.1은 **DeBERTa 계열 학습 시 step1부터 NaN 발산**(팀A 발견, 확인 완료).
평가 서버는 **transformers 4.46.3 / sklearn 1.8.0 / numpy 1.26.4 / torch 2.7.1+cu128**을 쓴다.
→ **모두 서버 버전에 맞춰 학습**한다(DeBERTa 버그 회피 + 제출 시 로드 정합).

```bash
git clone https://github.com/Multi-Bus/dacon-2026.git
cd dacon-2026

# torch는 인덱스 지정 설치 (드라이버가 cu128 미지원이면 2.5.1+cu121도 4.46.3과 호환)
pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_gpu.txt     # GPU 팀원
# 또는
pip install -r requirements_cpu.txt     # CPU 팀원
```

레포 구조:
```
dacon-2026/
├── dacon_data/data/     # train.jsonl(7만), train_labels.csv, test.jsonl
├── train/
│   ├── features.py, cv.py         # ★ 수정 금지 (전원 동일 fold 분할이 비교의 전제)
│   ├── bakeoff_fold0.py           # GPU 팀원 실행 (NaN 가드 + loss 로깅 내장)
│   ├── stack_v2.py                # CPU 팀원 실행
│   └── oof/                        # 기존 5뷰 OOF
├── requirements_gpu.txt / requirements_cpu.txt
```

## 1. 역할 분담 (전원 4.46.3 env)

| 담당 | 모델 | tag | 비고 |
|---|---|---|---|
| GPU 팀원 A | microsoft/mdeberta-v3-base | mdeberta | 5.12.1 NaN이었음 → 4.46.3에서 재학습 |
| GPU 팀원 B | snunlp/KR-ELECTRA-discriminator → kykim/electra-kor-base | kr_electra / kor_electra | 순차(같은 GPU 2개) |
| 나(로컬) | Korean DeBERTa 계열 | kodeberta | team-lucid는 4.46.3에서 토크나이저 재시도 |

각자 **fold0만** 학습(5-fold 아님). klue-base fold0=0.7143 대비 비교.

### 1) 스모크 먼저 (VRAM/속도 확인, 5분)
```bash
cd train
python bakeoff_fold0.py --model microsoft/mdeberta-v3-base --tag mdeberta --bs 8 --grad_ckpt --smoke
```
- VRAM 8GB급: `--bs 8 --grad_ckpt` / 12GB+: `--grad_ckpt` 빼고 `--bs 16` / OOM 시: `--bs 4 --accum 2`
- **loss가 NaN이면 즉시 raise됨**(스크립트에 가드 내장) → 조용한 낭비 없음. NaN 나면 transformers 버전부터 확인.

### 2) 본 실행 (fold0, 5 epoch)
```bash
nohup python bakeoff_fold0.py --model microsoft/mdeberta-v3-base --tag mdeberta --bs 8 --grad_ckpt > mdeberta.log 2>&1 &
```
team B(순차):
```bash
nohup python bakeoff_fold0.py --model snunlp/KR-ELECTRA-discriminator --tag kr_electra --bs 16 > kr_electra.log 2>&1 &
# ↑ 끝난 뒤 실행 (동시 실행 시 OOM)
nohup python bakeoff_fold0.py --model kykim/electra-kor-base --tag kor_electra --bs 16 > kor_electra.log 2>&1 &
```

### 3) 결과 확인
```bash
tail -8 <자기>.log
# [tag] fold0 F1=0.XXXX (klue-base fold0=0.709)  ← 이 줄이 핵심
```
결과 파일: `train/oof/<tag>_fold0.npz` (2~3MB) — 회수 대상.

## 2. CPU 팀원 C — 스택 재탐색 (GPU 불필요, 5~10분)
```bash
pip install -r requirements_cpu.txt
cd train
python stack_v2.py > stack_v2_result.log 2>&1
```
기존 5뷰로 forward-selection + isotonic 보정 A/B. **로그 전체를 공유**.

## 3. 결과 회수 — 각자 브랜치 push
```bash
git checkout -b bakeoff-<자기이름>
git add train/oof/<자기_tag>_fold0.npz          # CPU 팀원은 stack_v2_result.log
git commit -m "add <tag> fold0 OOF (transformers 4.46.3)"
git push origin bakeoff-<자기이름>
```
파일이 작아 push만으로 충분. push 후 알려주면 취합한다.

## 4. 하지 말 것
- `cv.py`/`features.py` 수정 금지 (동일 fold 분할).
- transformers **5.12.1 쓰지 말 것** — DeBERTa NaN. 반드시 4.46.3.
- `main` 직접 push 금지 — 각자 브랜치.
- `git add -A` 금지 — 위에 적힌 파일만 개별 add(대용량 실수 방지).

## 5. 배경 요약
- 스택 천장 = 5뷰 다 짜내도 0.7378. transformer 백본이 성능을 좌우(트리/선형은 보조).
- klue-large(같은 계열 대형)는 fold0 0.7117로 klue-base와 동일 수준 → 기각. **다른 아키텍처**가 답.
- mDeBERTa(다국어 DeBERTa-v3), 한국어 DeBERTa, ELECTRA 계열을 fold0로 빠르게 선별 → 이긴 것만 5-fold.
- 속도 제약이 없어졌으므로(예선은 성능만 봄) 무거운 후보도 성능만 높으면 채택 가능.
