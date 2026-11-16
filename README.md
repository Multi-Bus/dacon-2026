<div align="center">

# DACON 2026 다음 행동 예측

**대화 세션의 이전 맥락을 보고 다음 행동 14종 중 하나를 예측하는 모델 실험 저장소.** 평가 지표는 Macro-F1이다.

![transformers](https://img.shields.io/badge/transformers-4.46.3-FFD21E)
![torch](https://img.shields.io/badge/torch-2.7.1%2Bcu128-EE4C2C)
![Metric](https://img.shields.io/badge/metric-Macro--F1-6f42c1)

</div>

2026 AI·SW중심대학 디지털 경진대회 AI 부문(DACON) 출품작이다. 팀 Multi-Bus.

이 저장소에는 백본 비교와 스태킹 실험 코드, 뷰별 OOF 확률만 있다. 뷰별 모델 학습 코드와 제출 패키지는 `.gitignore`로 저장소에서 뺐다. 저장소만으로 최종 제출본을 다시 만들 수는 없다.

## 목차

- [기능](#기능)
- [동작 방식](#동작-방식)
- [빠른 시작](#빠른-시작)
- [사용 흐름](#사용-흐름)
- [설정](#설정)
- [결과](#결과)
- [한계](#한계)
- [내부 구조](#내부-구조)
- [개발](#개발)

## 기능

| 기능 | 내용 |
|---|---|
| 데이터 | 학습 7만 건(`train.jsonl`), 정답(`train_labels.csv`), 테스트 샘플 5건(`test.jsonl`)이 `dacon_data/data/`에 있다 |
| 피처 | 최근 history 턴을 텍스트로 직렬화하는 뷰와, 세션 메타와 action 시퀀스를 구조화 피처로 만드는 뷰를 `train/features.py`가 만든다 |
| 세션 단위 검증 | 같은 세션의 스텝이 여러 fold에 나뉘지 않도록 `StratifiedGroupKFold`로 나눈다 (`train/cv.py`) |
| 백본 비교 | 후보 백본을 fold 0에서만 학습해 기준 모델과 비교한다 (`train/bakeoff_fold0.py`) |
| 스태킹 | 뷰별 OOF 확률을 forward selection으로 고르고 로지스틱 회귀로 합친다. isotonic 보정 전후를 비교한다 (`train/stack_v2.py`) |
| OOF 확률 | 5개 뷰의 OOF 확률이 `train/oof/`에 있다 |

## 동작 방식

```
train.jsonl (세션 단위)
 ├─ 뷰 1  텍스트 피처 + 로지스틱 회귀 (m1v2)
 ├─ 뷰 2  구조화 피처 + LightGBM (m2v3)
 ├─ 뷰 3  구조화 피처 + CatBoost (m2cat)
 ├─ 뷰 4  KLUE RoBERTa 512토큰 파인튜닝 (m3_klue)
 └─ 뷰 5  action 시퀀스 원-핫 + 로지스틱 회귀 (s1)
        │
        ▼
  OOF 확률 → 로지스틱 회귀 스태킹 → 14클래스 예측
```

뷰 1, 2, 3, 5의 학습 코드는 저장소에 없고 OOF 파일만 있다. 뷰 4는 `bakeoff_fold0.py`로 fold 0만 재현할 수 있다.

최종 제출본은 스택 위에 클래스별 bias 14개를 더해 Macro-F1을 맞췄다. 이 bias 코드도 저장소에 없다.

`train/features.py`가 학습과 추론이 함께 쓰는 정본이다. 제출용 추론 스크립트(저장소 밖)에 같은 로직을 복사해 쓰므로 두 곳이 어긋나면 입력이 달라진다.

클래스는 14종이다: `read_file`, `grep_search`, `list_directory`, `glob_pattern`, `edit_file`, `write_file`, `apply_patch`, `run_bash`, `run_tests`, `lint_or_typecheck`, `ask_user`, `plan_task`, `web_search`, `respond_only`.

## 빠른 시작

GPU 학습에는 CUDA가 맞는 PyTorch가 필요하다.

```bash
git clone https://github.com/Multi-Bus/dacon-2026.git
cd dacon-2026

# GPU
pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_gpu.txt

# CPU
pip install -r requirements_cpu.txt
```

드라이버가 cu128을 지원하지 않으면 `torch==2.5.1+cu121`도 `transformers==4.46.3`과 호환된다.

## 사용 흐름

저장소 루트에서 두 단계로 돌린다.

1. **백본 선별 (GPU)**: 후보 백본을 fold 0에서만 학습해 기준 모델과 비교한다.

   ```bash
   cd train
   python bakeoff_fold0.py --model microsoft/mdeberta-v3-base --tag mdeberta --bs 8 --grad_ckpt --smoke
   python bakeoff_fold0.py --model microsoft/mdeberta-v3-base --tag mdeberta --bs 8 --grad_ckpt
   ```

   결과는 `train/oof/<tag>_fold0.npz`에 남는다. NaN이 나오면 스크립트가 즉시 중단한다.

2. **스태킹 (CPU)**: 기존 뷰와 신규 뷰를 forward selection으로 고르고, isotonic 보정 전후를 비교한다.

   ```bash
   cd train
   python stack_v2.py > stack_v2_result.log 2>&1
   ```

제출 패키지(추론 스크립트, 모델 파일, 앙상블 설정을 묶은 zip)는 저장소 밖에서 만든다. 제한은 zip 1GB 이하, 추론 10분 이내다.

## 설정

학습 버전은 평가 서버(transformers 4.46.3, scikit-learn 1.8.0, numpy 1.26.4)에 맞춘다.

| 패키지 | 버전 |
|---|---|
| torch | 2.7.1+cu128 (또는 2.5.1+cu121) |
| transformers | 4.46.3 |
| accelerate | 1.9.0 |
| sentencepiece | 0.1.99 |
| scikit-learn | 1.8.0 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| protobuf | 미고정 |

`transformers` 5.12.1에서는 DeBERTa 계열이 학습 첫 스텝부터 NaN으로 발산했다.

`train/features.py`와 `train/cv.py`는 모두가 같은 fold 분할을 쓰도록 바꾸지 않는다.

## 결과

| 항목 | 값 | 비고 |
|---|---|---|
| 기준 모델 fold 0 | 0.7143 | klue-base |
| 5뷰 스택 OOF (bias 전) | 0.7378 | 이 저장소의 `train/oof/`로 구성한 스택 |
| 5뷰 스택 OOF (bias 후) | 0.7404 | 팀 로컬 기록. 저장소에서 재현할 수 없다 |
| 리더보드 | 0.7386 | 2026-07-06 제출. 팀 제출 기록이며 저장소에는 없다 |

## 한계

- 최종 제출본의 학습 코드, 클래스 bias, 제출 패키지가 저장소에 없어 저장소만으로 재현할 수 없다.
- bias 후 OOF(0.7404)는 로컬 기록이고 저장소 코드로 확인할 수 없다.
- 클래스 bias는 검증 데이터에 맞춘 값이다. 리더보드와 OOF 차이(0.7386, 0.7404)가 크지 않았으나 과적합 가능성은 남는다.
- 뷰 1, 2, 3, 5는 OOF 확률만 있어 모델 설정을 코드로 확인할 수 없다.
- `test.jsonl`은 5건 샘플이라 저장소에서 추론 결과를 낼 수 없다.

<details>
<summary>미확인·제외 항목</summary>

미확인:

- 예선 최종 결과와 본선 진출 여부는 확인하지 않았다.
- 2026-07-06 이후 `stack7` 계열 제출의 점수는 확인하지 않았다.
- 학습 실행 시간과 GPU 사양은 기록하지 않았다.
- 지원하는 Python 버전은 확인하지 않았다.
- 이 문서를 쓰면서 학습을 다시 돌려 결과를 확인하지 않았다.
- 리더보드 점수와 OOF 점수가 최종 제출본과 같은 모델의 값인지 확인하지 않았다.

제외:

- 추론 속도 최적화는 하지 않았다. 제한(추론 10분)만 지킨다.

</details>

## 내부 구조

<details>
<summary>파일 구성</summary>

```
dacon-2026/
├── dacon_data/data/        train.jsonl(7만), train_labels.csv, test.jsonl(5건 샘플), sample_submission.csv
├── train/
│   ├── features.py         피처 정본. fold 분할 고정을 위해 수정하지 않는다
│   ├── cv.py               세션 단위 fold 분할. 수정하지 않는다
│   ├── bakeoff_fold0.py    백본 fold 0 학습 (NaN 가드 내장)
│   ├── stack_v2.py         스태킹과 isotonic 보정 비교
│   ├── prefetch_models.py  후보 백본 3종 가중치를 HF 캐시에 미리 받기
│   └── oof/                뷰별 OOF 확률(.npy)과 oof_ids.json
├── requirements_cpu.txt
└── requirements_gpu.txt
```

`.gitignore`가 화이트리스트 방식이다. 위 목록 밖의 파일(학습 코드, 제출 패키지, 모델 가중치)은 저장소에 올라가지 않는다.

</details>

<details>
<summary>팀 분업과 브랜치 규칙</summary>

- 담당별로 fold 0만 학습해 결과를 비교하고, 이긴 후보만 5-fold로 완주한다.
- 결과 파일은 `git add`로 개별 파일만 올린다. `git add -A`는 쓰지 않는다. 대용량 파일이 섞이는 것을 막기 위해서다.
- `main`에 직접 push하지 않는다. 담당자별 브랜치에 올린다.

</details>

## 개발

- 새 백본은 `bakeoff_fold0.py`로 먼저 비교한다.
- 새 뷰를 추가하면 `stack_v2.py`의 `CANDIDATES`에 넣고 다시 스택을 돌린다.
