#!/usr/bin/env python3
"""Tables for the paper from results/scores/sentences.csv (written by score.py).

    python code/analysis.py

Writes results/tables/*.md and results/tables/main_results.csv. Confidence
intervals are 95% percentile bootstrap intervals over sentences (1,000
resamples drawn with seed 42 for every interval, so that each interval is
reproducible on its own); differences between systems are paired.
"""
from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results" / "scores"
TABLES = ROOT / "results" / "tables"
OUTPUTS = ROOT / "results" / "outputs"
FLORES_REF = ROOT / "data" / "raw" / "flores200_dataset" / "devtest" / "eng_Latn.devtest"
FPS = 30.0
B = 1000

LABEL = {
    "E0_orig": "Input as is (lookup + fingerspelling)",
    "E1_wordnet": "WordNet substitution",
    "E2_embed": "Embedding substitution",
    "E3_simplify": "LLM simplification",
    "E4_hints": "LLM rewrite with hints",
    "E5_hints_constrained": "LLM rewrite with hints, restricted",
    "E6_simplify_constrained": "LLM simplification, restricted",
    "T0_nllb": "NLLB",
    "T1_eurollm": "EuroLLM",
    "T2_nllb_embed": "NLLB + embedding substitution",
    "T3_nllb_constrained": "NLLB + LLM rewrite with hints, restricted",
    "T4_eurollm_constrained": "EuroLLM, restricted",
    "T5_eurollm_hints_constrained": "EuroLLM with hints, restricted",
    "T6_nllb_simplify_constrained": "NLLB + LLM simplification, restricted",
    "E6nofilter_simplify_constrained": "LLM simplification, restricted, country names allowed",
    "R5_E6_simplify_constrained": "LLM simplification, restricted, best of 5",
    "R5_T6_nllb_simplify_constrained": "NLLB + LLM simplification, restricted, best of 5",
}
VOCAB_LABEL = {"is": "IS lexicon", "freq500": "Top 500", "freq1000": "Top 1,000",
               "freq2000": "Top 2,000", "freq5000": "Top 5,000"}


def base_name(system: str) -> str:
    return system.split("@")[0].split("__")[0]


def boot_ci(values: np.ndarray, stat=np.mean) -> tuple[float, float]:
    n = len(values)
    idx = np.random.default_rng(42).integers(0, n, size=(B, n))  # same resamples for every call
    s = np.array([stat(values[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def ratio_ci(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    n = len(num)
    idx = np.random.default_rng(42).integers(0, n, size=(B, n))
    s = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def corpus_scores(corpus: str, system: str, ids: list[int]) -> tuple[float, float]:
    """Corpus-level chrF and BLEU (sacreBLEU defaults) on the given FLORES sentences."""
    ref = FLORES_REF.read_text(encoding="utf-8").splitlines()
    recs = {}
    for line in (OUTPUTS / corpus / f"{system}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r["output"]
    ids = [i for i in sorted(ids) if i in recs]
    hyp, refs = [recs[i] for i in ids], [[ref[i] for i in ids]]
    return sacrebleu.corpus_chrf(hyp, refs).score, sacrebleu.corpus_bleu(hyp, refs).score


def paired_diff(df: pd.DataFrame, corpus: str, vocab: str, a: str, b: str, col: str) -> dict:
    """Mean of (a - b) over the sentences both systems were run on, with a paired
    95% percentile bootstrap interval."""
    ga = df[(df.corpus == corpus) & (df.vocab == vocab) & (df.system == a)].set_index("id")[col]
    gb = df[(df.corpus == corpus) & (df.vocab == vocab) & (df.system == b)].set_index("id")[col]
    common = ga.index.intersection(gb.index)
    d = (ga.loc[common] - gb.loc[common]).to_numpy(dtype=float)
    if len(d) == 0:
        return {}
    lo, hi = boot_ci(d)
    return {"n": len(d), "mean": float(d.mean()), "lo": lo, "hi": hi}


def fmt(x, ci=None, pct=False, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    k = 100 if pct else 1
    s = f"{k * x:.{nd}f}"
    if ci:
        s += f" ({k * ci[0]:.{nd}f}–{k * ci[1]:.{nd}f})"
    return s


def summarise_group(g: pd.DataFrame) -> dict:
    g = g.sort_values("id")
    d = {"n": len(g),
         "fs_rate": g["n_oov"].sum() / max(g["content"].sum(), 1),
         "fs_rate_ci": ratio_ci(g["n_oov"].to_numpy(), g["content"].to_numpy()),
         "letters": g["letters_oov"].mean(),
         "time_s": g["frames"].mean() / FPS,
         "time_ci": tuple(c / FPS for c in boot_ci(g["frames"].to_numpy())),
         "fs_share": g["fs_oov_frames"].sum() / max(g["frames"].sum(), 1),
         "letters_name": g["letters_name"].mean(),
         "signs": (g["n_sign"] + g["n_sign_fn"]).mean(),
         "latency": g["seconds"].mean(),
         "truncated": g["truncated"].mean(),
         "length_ratio": g["words_out"].sum() / max(g["words_in"].sum(), 1)}
    if "comet" in g and g["comet"].notna().all() and len(g):
        d["comet"] = g["comet"].mean()
        d["comet_ci"] = boot_ci(g["comet"].to_numpy())
    if "chrf" in g and g["chrf"].notna().all() and len(g):
        d["chrf_sent"] = g["chrf"].mean()
    if "cosine" in g and g["cosine"].notna().all() and len(g):
        d["cosine"] = g["cosine"].mean()
        d["entail"] = g["mutual_entailment"].mean()
        d["entail_ci"] = boot_ci(g["mutual_entailment"].to_numpy())
    return d


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORES / "sentences.csv")
    df["base"] = df["system"].map(base_name)
    df = df[~df["system"].str.contains("@")]  # main model only
    import json
    m = json.loads((ROOT / "data" / "manifest.json").read_text())
    llm_ids = set(m["flores_llm_ids"])
    df_all = df
    # Comparisons between systems use the sentences every system was run on:
    # the 100-sentence FLORES subset and the 200 AMI segments.
    in_subset = df["id"].astype(str).map(lambda x: x.isdigit() and int(x) in llm_ids)
    df_cmp = df[(~df["corpus"].str.startswith("flores")) | in_subset]

    # ---- Table A: how much a dictionary avatar fingerspells, by vocabulary ----
    rows = []
    for corpus, sysn, label in [("ami", "E0_orig", "AMI meetings (English)"),
                                ("flores_eng", "E0_orig", "FLORES (English)"),
                                ("flores_ell_Grek", "T0_nllb", "FLORES Greek→English (NLLB)"),
                                ("flores_fra_Latn", "T0_nllb", "FLORES French→English (NLLB)")]:
        for v in ["is", "freq500", "freq1000", "freq2000", "freq5000"]:
            g = df_all[(df_all.corpus == corpus) & (df_all.system == sysn) & (df_all.vocab == v)]
            if g.empty:
                continue
            s = summarise_group(g)
            rows.append(f"| {label} | {VOCAB_LABEL[v]} | {fmt(s['fs_rate'], s['fs_rate_ci'], pct=True)} | "
                        f"{fmt(s['letters'])} | {fmt(s['time_s'], s['time_ci'])} | {fmt(s['fs_share'], pct=True)} |")
    head = ("| Text | Vocabulary | Content words fingerspelled, % (95% CI) | Fingerspelled letters per "
            "sentence | Estimated signing time per sentence, s (95% CI) | Share of signing time spent "
            "fingerspelling, % |\n| --- | --- | --- | --- | --- | --- |")
    (TABLES / "tableA_problem.md").write_text(head + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    # ---- main results per corpus x vocabulary x system ----
    main_rows = []
    for (corpus, v, system), g in df_cmp.groupby(["corpus", "vocab", "system"]):
        s = summarise_group(g)
        s.update(corpus=corpus, vocab=v, system=system, base=base_name(system))
        if corpus.startswith("flores"):
            s["chrf"], s["bleu"] = corpus_scores(corpus, system, [int(i) for i in g["id"]])
        main_rows.append(s)
    res = pd.DataFrame(main_rows)
    res.to_csv(TABLES / "main_results.csv", index=False)

    def table(corpora, systems, title_file, meaning_cols):
        lines = []
        for v in ["is", "freq1000", "freq5000"]:
            for corpus in corpora:
                base = None
                for sysn in systems:
                    name = sysn if sysn in ("E0_orig", "E3_simplify", "T0_nllb", "T1_eurollm") else f"{sysn}__{v}"
                    r = res[(res.corpus == corpus) & (res.vocab == v) & (res.system == name)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    if base is None:
                        base = r
                    dt = (r["time_s"] / base["time_s"] - 1) if base is not None else np.nan
                    cells = [VOCAB_LABEL[v], corpus.replace("flores_", "FLORES ").replace("_Latn", "").replace("_Grek", ""),
                             LABEL[base_name(name)], fmt(r["fs_rate"], r["fs_rate_ci"], pct=True),
                             fmt(r["letters"]), fmt(r["time_s"]), fmt(dt, pct=True, nd=0)]
                    if "comet" in meaning_cols:
                        cells += [fmt(r.get("comet"), r.get("comet_ci") if isinstance(r.get("comet_ci"), tuple) else None, nd=3),
                                  fmt(r.get("chrf"))]
                    if "entail" in meaning_cols:
                        cells += [fmt(r.get("cosine"), nd=3),
                                  fmt(r.get("entail"), r.get("entail_ci") if isinstance(r.get("entail_ci"), tuple) else None, pct=True)]
                    cells += [fmt(r["latency"], nd=2), fmt(r["truncated"], pct=True)]
                    lines.append("| " + " | ".join(cells) + " |")
        hdr = ["Vocabulary", "Text", "System", "Content words fingerspelled, % (95% CI)",
               "Letters per sentence", "Signing time, s", "Change in signing time"]
        if "comet" in meaning_cols:
            hdr += ["COMET (95% CI)", "chrF"]
        if "entail" in meaning_cols:
            hdr += ["Cosine similarity to input", "Mutual entailment with input, % (95% CI)"]
        hdr += ["Latency, s (2 CPU threads)", "Truncated, %"]
        txt = "| " + " | ".join(hdr) + " |\n| " + " | ".join("---" for _ in hdr) + " |\n" + "\n".join(lines) + "\n"
        (TABLES / title_file).write_text(txt, encoding="utf-8")

    en_sys = ["E0_orig", "E1_wordnet", "E2_embed", "E3_simplify", "E4_hints", "E6_simplify_constrained",
              "E5_hints_constrained", "E6nofilter_simplify_constrained", "R5_E6_simplify_constrained"]
    table(["flores_eng"], en_sys, "tableB_english_flores.md", ["comet"])
    table(["ami"], en_sys, "tableC_english_ami.md", ["entail"])
    tr_sys = ["T0_nllb", "T1_eurollm", "T2_nllb_embed", "T6_nllb_simplify_constrained",
              "R5_T6_nllb_simplify_constrained", "T4_eurollm_constrained"]
    table(["flores_ell_Grek", "flores_fra_Latn", "flores_spa_Latn", "flores_deu_Latn"], tr_sys,
          "tableD_translation.md", ["comet"])
    # ---- paired differences between systems (same sentences) ----
    pairs = []
    for v in ["is", "freq1000", "freq5000"]:
        pairs += [("flores_eng", v, f"E2_embed__{v}", f"E1_wordnet__{v}", "comet"),
                  ("flores_eng", v, f"E6_simplify_constrained__{v}", f"E2_embed__{v}", "comet"),
                  ("ami", v, f"E2_embed__{v}", f"E1_wordnet__{v}", "mutual_entailment"),
                  ("ami", v, f"E6_simplify_constrained__{v}", f"E2_embed__{v}", "mutual_entailment")]
        for c in ["flores_ell_Grek", "flores_fra_Latn"]:
            pairs += [(c, v, f"T4_eurollm_constrained__{v}", f"T6_nllb_simplify_constrained__{v}", "comet"),
                      (c, v, f"T4_eurollm_constrained__{v}", f"T6_nllb_simplify_constrained__{v}", "chrf"),
                      (c, v, f"T6_nllb_simplify_constrained__{v}", f"T2_nllb_embed__{v}", "comet")]
    pairs += [("flores_eng", "is", "R5_E6_simplify_constrained__is", "E6_simplify_constrained__is", "comet"),
              ("flores_eng", "is", "E6_simplify_constrained__is", "E6nofilter_simplify_constrained__is", "comet"),
              ("flores_eng", "is", "E6_simplify_constrained__is", "E5_hints_constrained__is", "comet"),
              ("flores_eng", "is", "E4_hints__is", "E3_simplify", "comet")]
    lines = []
    for corpus, v, a, b, col in pairs:
        d = paired_diff(df_cmp, corpus, v, a, b, col)
        if not d:
            continue
        k = 100 if col == "mutual_entailment" else 1
        nd = 1 if col in ("mutual_entailment", "chrf") else 3
        lines.append(f"| {corpus} | {VOCAB_LABEL[v]} | {a.split('__')[0]} − {b.split('__')[0]} | {col} | {d['n']} | "
                     f"{k * d['mean']:+.{nd}f} ({k * d['lo']:+.{nd}f} to {k * d['hi']:+.{nd}f}) |")
    (TABLES / "paired_differences.md").write_text(
        "| Text | Vocabulary | Difference | Measure | Sentences | Mean difference (95% CI) |\n"
        "| --- | --- | --- | --- | --- | --- |\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print((TABLES / "tableA_problem.md").read_text())
    print((TABLES / "paired_differences.md").read_text())


if __name__ == "__main__":
    main()
