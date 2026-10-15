#!/usr/bin/env python3
"""Score every output file: signing plan, meaning preservation and latency.

    python code/score.py            # writes results/scores/sentences.csv and summary.csv

Signing metrics (per sentence, for the system's vocabulary; vocabulary-free
systems are scored against every vocabulary):
  content (signed or fingerspelled content words), n_sign, n_sign_fn (signed stop words,
  pronouns and interjections, not content words), n_oov (avoidable fingerspelling), n_name, letters_oov,
  letters_name, frames (estimated signing time at the IS clip lengths).
Meaning:
  FLORES : chrF and BLEU against the English reference (sacreBLEU), COMET
           (wmt22-comet-da, from results/comet, written by comet_score.py).
  AMI    : cosine similarity of sentence embeddings (all-mpnet-base-v2) and
           mutual entailment (cross-encoder/nli-deberta-v3-base) with the input.
Latency: generation seconds on 2 CPU threads, plus the upstream NLLB seconds
for cascades.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from run import OUT, PER_CLIP, inputs, manifest, read_jsonl, VOCABS_ALL  # noqa: E402
from sgt.signing import plan  # noqa: E402
from sgt.vocab import get_vocabulary  # noqa: E402

SCORES = ROOT / "results" / "scores"
COMET = ROOT / "results" / "comet"
FPS = 30.0  # frame rate assumed for the clips of the IS dictionary, which the hand-tracking files do not record
# (five source clips that were checked ran at about 30 fps)


def system_vocab(system: str):
    base = system.split("@")[0]
    return base.split("__")[1] if "__" in base else None


def load_ami_meaning_models():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-base", device="cpu")
    return emb, nli


def nli_labels(nli) -> list[str]:
    cfg = nli.model.config
    return [cfg.id2label[i].lower() for i in range(len(cfg.id2label))]


def ami_meaning(pairs, emb, nli, cache_f: Path) -> dict:
    """pairs: list of (key, input, output) -> {key: (cosine, mutual_entailment)}"""
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    todo = [p for p in pairs if p[0] not in cache]
    if todo:
        a = emb.encode([p[1] for p in todo], batch_size=32, normalize_embeddings=True)
        b = emb.encode([p[2] for p in todo], batch_size=32, normalize_embeddings=True)
        cos = (a * b).sum(axis=1)
        labels = nli_labels(nli)
        ent = labels.index("entailment")
        fwd = nli.predict([(p[1], p[2]) for p in todo], batch_size=32, apply_softmax=True)
        bwd = nli.predict([(p[2], p[1]) for p in todo], batch_size=32, apply_softmax=True)
        for p, c, f, bk in zip(todo, cos, fwd, bwd):
            cache[p[0]] = [float(c), int(np.argmax(f) == ent and np.argmax(bk) == ent),
                           float(f[ent]), float(bk[ent])]
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps(cache))
    return cache


def main():
    SCORES.mkdir(parents=True, exist_ok=True)
    m = manifest()
    sample = {"flores": set(m["flores_ids"]), "ami": set(m["ami_ids"])}
    vocabs = {v: get_vocabulary(v, PER_CLIP) for v in VOCABS_ALL}
    refs = {x["id"]: x["ref"] for x in inputs("flores_eng")}
    ami_inputs = {x["id"]: x["text"] for x in inputs("ami")}
    rows = []
    corpus_level = {}
    emb = nli = None
    for corpus_dir in sorted(OUT.iterdir()):
        corpus = corpus_dir.name
        kind = "ami" if corpus == "ami" else "flores"
        for f in sorted(corpus_dir.glob("*.jsonl")):
            if "@test" in f.name or re.match(r"S\d+_", f.stem):
                continue  # test runs; best-of-N candidate files (scored through their R file)
            system = f.stem
            only = os.environ.get("SCORE_ONLY")
            if only and only not in f"{corpus}/{system}":
                continue
            recs = [r for r in read_jsonl(f) if r["id"] in sample[kind]]
            if not recs:
                continue
            complete = len(recs) == len(sample[kind])
            v_sys = system_vocab(system)
            vlist = [v_sys] if v_sys else VOCABS_ALL
            comet = {}
            cf = COMET / corpus / f"{system}.json"
            if kind == "flores" and cf.exists():
                comet = json.loads(cf.read_text())
            meaning = {}
            if kind == "ami":
                if emb is None:
                    emb, nli = load_ami_meaning_models()
                pairs = [(str(r["id"]), ami_inputs[r["id"]], r["output"]) for r in recs]
                meaning = ami_meaning(pairs, emb, nli, SCORES / "ami_meaning_cache" / f"{system}.json")
            if kind == "flores":
                hyp = [r["output"] for r in recs]
                ref = [[refs[r["id"]] for r in recs]]
                corpus_level[(corpus, system)] = (sacrebleu.corpus_chrf(hyp, ref).score,
                                                  sacrebleu.corpus_bleu(hyp, ref).score)
            for r in recs:
                out = r["output"]
                base = {"corpus": corpus, "system": system, "id": r["id"], "complete": complete,
                        "seconds": float(r.get("seconds", 0.0)) + float(r.get("upstream_seconds", 0.0)),
                        "llm_seconds": float(r.get("seconds", 0.0)) if "prompt_tokens" in r else 0.0,
                        "truncated": int(r.get("finish") == "length"),
                        "skipped": int(bool(r.get("skipped"))),
                        "words_in": len(r["input"].split()), "words_out": len(out.split())}
                if kind == "flores":
                    base["chrf"] = sacrebleu.sentence_chrf(out, [refs[r["id"]]]).score
                    base["comet"] = comet.get(str(r["id"]), np.nan)
                else:
                    c = meaning.get(str(r["id"]))
                    base["cosine"], base["mutual_entailment"] = (c[0], c[1]) if c else (np.nan, np.nan)
                for v in vlist:
                    p = plan(out, vocabs[v])
                    rows.append({**base, "vocab": v, "content": p.content, "n_sign": p.n_sign,
                                 "n_sign_fn": p.n_sign_fn,
                                 "n_oov": p.n_oov, "n_name": p.n_name, "letters_oov": p.letters_oov,
                                 "letters_name": p.letters_name, "frames": p.frames,
                                 "fs_frames": (p.letters_oov + p.letters_name) * vocabs[v].durations.letter,
                                 "fs_oov_frames": p.letters_oov * vocabs[v].durations.letter,
                                 "oov_words": " ".join(p.oov_words)})
            print(corpus, system, len(recs), "complete" if complete else "PARTIAL", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(SCORES / "sentences.csv", index=False)
    summarise(df, corpus_level)


def summarise(df: pd.DataFrame, corpus_level: dict):
    out = []
    for (corpus, system, vocab), g in df.groupby(["corpus", "system", "vocab"]):
        d = {"corpus": corpus, "system": system, "vocab": vocab, "n": len(g),
             "complete": bool(g["complete"].all()),
             "fs_rate": g["n_oov"].sum() / max(g["content"].sum(), 1),
             "oov_per_sent": g["n_oov"].mean(),
             "letters_oov_per_sent": g["letters_oov"].mean(),
             "signs_per_sent": (g["n_sign"] + g["n_sign_fn"]).mean(),
             "seconds_signing": g["frames"].mean() / FPS,
             "share_time_fingerspelling": g["fs_oov_frames"].sum() / max(g["frames"].sum(), 1),
             "share_time_fingerspelling_incl_names": g["fs_frames"].sum() / max(g["frames"].sum(), 1),
             "latency_s": g["seconds"].mean(), "latency_p95_s": g["seconds"].quantile(0.95),
             "truncated": g["truncated"].mean(), "skipped": g["skipped"].mean(),
             "length_ratio": g["words_out"].sum() / max(g["words_in"].sum(), 1)}
        if "chrf" in g and g["chrf"].notna().any():
            d["chrf"], d["bleu"] = corpus_level.get((corpus, system), (np.nan, np.nan))
            d["chrf_sent_mean"] = g["chrf"].mean()
            d["comet"] = g["comet"].mean() if g["comet"].notna().all() else np.nan
        if "cosine" in g and g["cosine"].notna().any():
            d["cosine"] = g["cosine"].mean()
            d["mutual_entailment"] = g["mutual_entailment"].mean()
        out.append(d)
    s = pd.DataFrame(out)
    s.to_csv(SCORES / "summary.csv", index=False)
    print(s.to_string(max_rows=400, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
