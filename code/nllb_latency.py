#!/usr/bin/env python3
"""NLLB latency when sentences are translated one at a time (batch size 1).

    python code/nllb_latency.py     # -> results/tables/nllb_latency.csv

The T0 outputs were produced in batches of 16, and their time per sentence is the
batch time divided by the batch size. This script measures the latency of
translating one sentence at a time (2 CPU threads, beam size 4) on the 100
sentences used for the LLM systems, and checks that the translations are the same.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from run import NLLB_DIR, OUT, SRC_LANGS, inputs, read_jsonl  # noqa: E402
from sgt.nllb import NLLB  # noqa: E402


def main():
    tr = NLLB(NLLB_DIR, threads=2)
    rows = []
    for lang in SRC_LANGS:
        items = inputs(f"flores_{lang}", llm=True)
        batch = {r["id"]: r["output"] for r in read_jsonl(OUT / f"flores_{lang}" / "T0_nllb.jsonl")}
        tr.translate([items[0]["text"]], lang, batch=1)  # warm-up
        same = 0
        for x in items:
            t0 = time.perf_counter()
            hyp = tr.translate([x["text"]], lang, batch=1)[0][0]
            dt = time.perf_counter() - t0
            same += hyp == batch.get(x["id"])
            rows.append({"lang": lang, "id": x["id"], "seconds": dt})
        d = pd.DataFrame([r for r in rows if r["lang"] == lang])
        print(lang, f"mean {d.seconds.mean():.3f} s, p95 {d.seconds.quantile(0.95):.3f} s, "
                    f"identical to batched output: {same}/{len(items)}", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables" / "nllb_latency.csv"
    df.groupby("lang")["seconds"].agg(["mean", "median", lambda s: s.quantile(0.95)]).rename(
        columns={"<lambda_0>": "p95"}).to_csv(out)
    df.to_csv(ROOT / "results" / "scores" / "nllb_latency_sentences.csv", index=False)
    print(json.dumps(df.groupby("lang")["seconds"].mean().round(3).to_dict()))


if __name__ == "__main__":
    main()
