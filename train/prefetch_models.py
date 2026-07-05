"""후보 백본 가중치 사전 다운로드(네트워크만, GPU 미사용).

klue-large 학습과 충돌 없이 HF 캐시에 받아둔다. 실패한 모델은 스킵하고 계속.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = [
    "microsoft/mdeberta-v3-base",
    "snunlp/KR-ELECTRA-discriminator",
    "kykim/electra-kor-base",
]
for m in MODELS:
    try:
        print(f"[prefetch] {m} ...", flush=True)
        AutoTokenizer.from_pretrained(m)
        AutoModelForSequenceClassification.from_pretrained(m, num_labels=14)
        print(f"[prefetch] {m} OK", flush=True)
    except Exception as e:
        print(f"[prefetch] {m} FAIL: {type(e).__name__}: {e}", flush=True)
print("[prefetch] 완료", flush=True)
