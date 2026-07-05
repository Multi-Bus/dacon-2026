"""백본 후보 fold0 게이트 — 임의 transformer를 klue와 동일 조건으로 fold0만 학습.

klue-base fold0=0.709 / klue-large fold0(진행 중) 대비 비교.
8GB 대응: grad ckpt + bf16 + 사전토큰화. fold0 확률을 oof/{tag}_fold0.npz 저장(스택 다양성 측정용).
동일 GroupKFold fold0 → 다른 후보와 공정 비교.

사용: python bakeoff_fold0.py --model microsoft/mdeberta-v3-base --tag mdeberta --bs 8
스모크(VRAM/속도 실측 후 종료): python bakeoff_fold0.py --model ... --tag ... --bs 8 --smoke
"""
from __future__ import annotations

import os, sys, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F
from cv import make_group_folds
from sklearn.metrics import f1_score

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "dacon_data", "data")
OOF_DIR = os.path.join(HERE, "oof")
MAXLEN, EPOCHS, LR = 512, 5, 2e-5


def meta_prefix(s):
    m = F.meta_dict(s)
    return (f"[TIER]{m['user_tier']} [LANG]{m['language_pref']} "
            f"[CI]{m['last_ci']} [TOPLANG]{m['top_lang']} "
            f"[LAST]{m['last_action']} [LAST2]{m['last2_action']} "
            f"[TURN]{m['turn_index']} [DIRTY]{m['git_dirty']} ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)

    samples = F.load_jsonl(os.path.join(DATA, "train.jsonl"))
    labels = F.load_labels(os.path.join(DATA, "train_labels.csv"))
    ids = [s["id"] for s in samples]
    y = np.array([labels[i] for i in ids])
    X = [meta_prefix(s) + F.build_text(s) for s in samples]
    label2id = {c: i for i, c in enumerate(F.ACTIONS)}
    n_cls = len(F.ACTIONS)
    print(f"[{args.tag}] model={args.model} bs={args.bs}x{args.accum} | {torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    ENC = [tok(x, truncation=True, max_length=MAXLEN)["input_ids"] for x in X]

    class DS(Dataset):
        def __init__(s, idxs, lab=True): s.idxs, s.lab = idxs, lab
        def __len__(s): return len(s.idxs)
        def __getitem__(s, j):
            i = s.idxs[j]; it = {"input_ids": ENC[i]}
            if s.lab: it["labels"] = label2id[y[i]]
            return it

    def collate(batch):
        labs = torch.tensor([b.pop("labels") for b in batch]) if "labels" in batch[0] else None
        out = tok.pad(batch, return_tensors="pt")
        if labs is not None: out["labels"] = labs
        return out

    folds = make_group_folds(ids, y, n_splits=5)
    tr, va = folds[0]
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_cls,
        id2label={i: c for c, i in label2id.items()}, label2id=label2id).cuda()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(); model.config.use_cache = False
    dl = DataLoader(DS(tr), batch_size=args.bs, shuffle=True, collate_fn=collate)
    vdl = DataLoader(DS(va, False), batch_size=16, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    tot = (len(dl) // args.accum) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * tot), tot)
    step = 0
    for ep in range(EPOCHS):
        model.train(); t = time.time(); opt.zero_grad()
        for bi, b in enumerate(dl):
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            lb = b.pop("labels")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**b).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), lb)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[{args.tag}] step {step}에서 loss={loss.item()} (NaN/Inf) — 학습 발산. "
                    "DeBERTa 계열은 transformers 5.12.1에서 backward가 깨짐 → 4.46.3 사용 필요. 즉시 중단.")
            (loss / args.accum).backward()
            if (bi + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
                step += 1
                if step % 200 == 1:
                    print(f"      step {step} loss={loss.item():.4f}", flush=True)
                if args.smoke and step >= 60:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    sec = (time.time() - t) / step
                    est = sec * (len(dl) // args.accum) / 60
                    print(f"[SMOKE] 60스텝 OK | peak VRAM {mem:.2f}GB | "
                          f"{sec:.2f}s/step | 예상 {est:.0f}min/epoch (x5 epoch = {est*5:.0f}min)", flush=True)
                    return
        print(f"    ep{ep+1}/{EPOCHS} {(time.time()-t)/60:.1f}min", flush=True)
    model.eval(); probs = []
    with torch.inference_mode():
        for b in vdl:
            b = {k: v.cuda() for k, v in b.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**b).logits
            probs.append(torch.softmax(lg.float(), -1).cpu().numpy())
    vp = np.vstack(probs)
    f1 = f1_score(y[va], np.array(F.ACTIONS)[vp.argmax(1)], labels=F.ACTIONS, average="macro", zero_division=0)
    print(f"[{args.tag}] fold0 F1={f1:.4f}  (klue-base fold0=0.709)", flush=True)
    np.savez(os.path.join(OOF_DIR, f"{args.tag}_fold0.npz"),
             probs=vp, va=np.array(va), classes=np.array(F.ACTIONS))
    print(f"[{args.tag}] 저장: oof/{args.tag}_fold0.npz", flush=True)


if __name__ == "__main__":
    main()
