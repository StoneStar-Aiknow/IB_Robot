"""Train and export the L3 semantic gate model (Python 3.10+, numpy).

The artefact is plain JSON so the agent can score it with nothing but the
standard library (see ``score_text.py``); training is the only part that needs
numpy. Hash and n-grams are fixed in the artefact, so a model trained here and
a model trained on the board score identically.

Corpus format: one example per line, either ``<text> 是`` / ``<text> 不是``
(是 = addressed to the robot) or ``{"text": ..., "label": 0|1}``. Usage:

    python3 tools/addressee/train_model.py --corpus corpus.txt \
        --out models/addressee/addressee_zh_v1.json --epochs 120

Cross-validation groups near-duplicate examples (character 3-gram Jaccard >=
0.7) so paraphrases cannot leak across folds. Full-corpus training follows the
same seeds, which is why the shipped artefact and the CV numbers agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter

import numpy as np

DIM = 1 << 16
NGRAMS = (1, 2, 3)
WAKE_WORDS = (
    "灵犀灵犀灵犀", "灵犀灵犀", "灵犀灵溪", "灵犀", "灵溪", "林夕", "林星",
    "小欧小欧", "小欧", "小鸥", "小猴", "小猫", "小小",
)
PUNCT = re.compile(r"[\s，。！？、,.!?~～\"'“”‘’()（）\[\]【】:：;；\-—…]+")


def fnv1a32(text: str) -> int:
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def normalize(text: str) -> str:
    return PUNCT.sub("", text.strip().lower())


def strip_wake(text: str) -> str:
    for _ in range(2):
        for wake in WAKE_WORDS:
            if text.startswith(wake) and len(text) > len(wake):
                text = text[len(wake):]
                break
        else:
            break
    return text


def features(text: str) -> Counter[int]:
    body = normalize(strip_wake(normalize(text)))
    padded = f"<{body}>"
    out: Counter[int] = Counter()
    for n in NGRAMS:
        for i in range(len(padded) - n + 1):
            out[fnv1a32(padded[i:i + n]) % DIM] += 1
    return out


def vectorize(text: str) -> tuple[np.ndarray, np.ndarray]:
    counts = features(text)
    idx = np.array(list(counts.keys()), dtype=np.int64)
    val = np.array([1.0 + math.log(c) for c in counts.values()], dtype=np.float32)
    val /= math.sqrt(float((val ** 2).sum())) or 1.0
    return idx, val


def load_corpus(path: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            rec = json.loads(line)
            rows.append((str(rec["text"]).strip(), int(rec["label"])))
            continue
        match = re.match(r"^(.*?)\s+(是|不是)$", line)
        if match:
            rows.append((match.group(1).strip(), 1 if match.group(2) == "是" else 0))
    return rows


def near_duplicate_clusters(texts: list[str], threshold: float = 0.7) -> list[int]:
    sets = []
    for text in texts:
        body = normalize(strip_wake(normalize(text)))
        sets.append({body[i:i + 3] for i in range(max(1, len(body) - 2))})
    parent = list(range(len(texts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    buckets: dict[str, list[int]] = {}
    for i, gramset in enumerate(sets):
        for gram in list(gramset)[:40]:
            for j in buckets.get(gram, ())[:200]:
                inter = len(gramset & sets[j])
                if inter and inter / len(gramset | sets[j]) >= threshold:
                    parent[find(i)] = find(j)
            buckets.setdefault(gram, []).append(i)
    return [find(i) for i in range(len(texts))]


def fit(vecs, labels, train_idx, epochs=120, lr=0.3, seed=0):
    rng = random.Random(seed)
    w = np.zeros(DIM, dtype=np.float32)
    m = np.zeros(DIM, dtype=np.float32)
    v = np.zeros(DIM, dtype=np.float32)
    bias = 0.0
    pos_weight = float((labels[train_idx] == 0).sum()) / max(1.0, float((labels[train_idx] == 1).sum()))
    for epoch in range(1, epochs + 1):
        order = list(train_idx)
        rng.shuffle(order)
        for i in order:
            idx, val = vecs[i]
            z = float(w[idx] @ val) + bias
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            grad = (p - labels[i]) * (pos_weight if labels[i] == 1 else 1.0)
            g = val * grad
            m[idx] = 0.9 * m[idx] + 0.1 * g
            v[idx] = 0.999 * v[idx] + 0.001 * g * g
            step = lr / (1.0 + 0.002 * epoch)
            w[idx] -= step * m[idx] / (np.sqrt(v[idx]) + 1e-8) + 1e-5 * w[idx]
            bias -= step * grad
    return w, bias


def predict_scores(w, bias, vecs) -> np.ndarray:
    out = np.zeros(len(vecs), dtype=np.float32)
    for i, (idx, val) in enumerate(vecs):
        out[i] = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(w[idx] @ val) + bias))))
    return out


def metrics(scores, labels, thr: float) -> dict:
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    return {
        "threshold": round(float(thr), 3),
        "accuracy": round((tp + tn) / len(labels), 4),
        "recall": round(tp / max(1, tp + fn), 4),
        "precision": round(tp / max(1, tp + fp), 4),
        "specificity": round(tn / max(1, tn + fp), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_corpus(args.corpus)
    texts = [t for t, _ in rows]
    labels = np.array([y for _, y in rows], dtype=np.float32)
    vecs = [vectorize(t) for t in texts]
    clusters = near_duplicate_clusters(texts)
    unique = sorted(set(clusters))
    random.Random(args.seed).shuffle(unique)
    fold_of = {cid: i % args.folds for i, cid in enumerate(unique)}
    fold = np.array([fold_of[c] for c in clusters])

    scores = np.zeros(len(labels), dtype=np.float32)
    for k in range(args.folds):
        train = np.where(fold != k)[0]
        test = np.where(fold == k)[0]
        w, bias = fit(vecs, labels, train, epochs=args.epochs, seed=args.seed + k)
        scores[test] = predict_scores(w, bias, [vecs[i] for i in test])

    np.savez_compressed(args.out + ".cv.npz", scores=scores, labels=labels, fold=fold)
    table = {f"{t:.2f}": metrics(scores, labels, t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)}
    print(f"corpus={len(rows)} pos={int(labels.sum())} neg={int(len(labels)-labels.sum())} clusters={len(unique)}")
    for key, row in table.items():
        print(f"  thr={key} " + json.dumps(row, ensure_ascii=False))

    w, bias = fit(vecs, labels, np.arange(len(labels)), epochs=args.epochs, seed=args.seed)
    nz = np.nonzero(w)[0]
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "feature": {"ngrams": list(NGRAMS), "dim": DIM, "hash": "fnv1a32", "pad": "<>", "normalize": "lower+strip-punct"},
        "wake_words": list(WAKE_WORDS),
        "bias": round(float(bias), 6),
        "weights": [[int(i), round(float(w[i]), 6)] for i in nz.tolist()],
        "training": {
            "epochs": args.epochs, "seed": args.seed, "folds": args.folds,
            "corpus_sha256": hashlib.sha256(open(args.corpus, "rb").read()).hexdigest(),
            "n": len(rows), "pos": int(labels.sum()), "neg": int(len(labels) - labels.sum()),
        },
        "cross_validation": table,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"wrote {args.out} weights={len(nz)} bias={bias:.4f} size={len(json.dumps(payload))/1024:.0f}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
