#!/usr/bin/env python3
"""Pick, for each sentence, the restricted candidate closest in meaning to the text
it rewrites (cosine similarity of all-mpnet-base-v2 embeddings).

    python code/rerank.py      # S4_<system>__<vocab>.jsonl -> R5_<system>__<vocab>.jsonl

Candidates: the greedy restricted output plus N sampled ones (run.py sample).
Latency counts every generation (run one after the other on the CPU).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from run import OUT, append, read_jsonl  # noqa: E402


def main():
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    for f in sorted(OUT.glob("*/S*_*.jsonl")):
        n_cand = int(f.stem.split("_")[0][1:]) + 1
        out = f.with_name("R" + str(n_cand) + f.stem[f.stem.index("_"):] + ".jsonl")
        out.unlink(missing_ok=True)
        for r in read_jsonl(f):
            t0 = time.perf_counter()
            src = r["rewrite_input"]
            c = r["candidates"]
            if len(c) == 1:
                best, sims = c[0], [1.0]
            else:
                e = emb.encode([src] + c, normalize_embeddings=True)
                sims = (e[1:] @ e[0]).tolist()
                best = c[max(range(len(c)), key=lambda i: sims[i])]
            dt = time.perf_counter() - t0
            append(out, {"id": r["id"], "input": r["input"], "output": best,
                         "seconds": sum(r["seconds"]) + dt, "upstream_seconds": r.get("upstream_seconds", 0.0),
                         "skipped": r.get("skipped", False), "candidate_similarity": sims,
                         "chosen": c.index(best)})
        print(f.parent.name, out.name, flush=True)


if __name__ == "__main__":
    main()
