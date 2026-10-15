#!/usr/bin/env python3
"""How much fingerspelling would disappear if the IS lexicon grew?

The IS lexicon is extended with the N most frequent English content lemmas it
lacks (wordfreq ranking, the same list as the freqK vocabularies; country names
skipped), and the input texts are re-planned without any rewriting.

    python code/lexicon_growth.py    # -> results/tables/lexicon_growth.md / .csv
    python code/lexicon_growth.py --list-only   # only the ranked list of missing lemmas

The ranked list of the lemmas the IS lexicon lacks (results/tables/missing_lemmas_is.csv)
is the order in which they are added; it is not filtered (it contains, for example,
profanity and lemmatisation errors of the frequency list).

Texts: all 2,162 AMI segments that pass the filters and all 1,012 FLORES
devtest English sentences.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from run import PER_CLIP, inputs  # noqa: E402
from sgt.signing import plan  # noqa: E402
from sgt.vocab import get_vocabulary, is_country, ranked_content_lemmas  # noqa: E402

FPS = 30.0
STEPS = [0, 50, 100, 200, 500, 1000, 2000]


def main():
    base = get_vocabulary("is", PER_CLIP)
    ranked = [(w, f) for w, f in ranked_content_lemmas() if w not in base.lemmas and not is_country(w)]
    missing = [w for w, _ in ranked]
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"rank": i + 1, "lemma": w, "summed_wordfreq_frequency": f}
                  for i, (w, f) in enumerate(ranked[:STEPS[-1]])]).to_csv(out / "missing_lemmas_is.csv", index=False)
    if "--list-only" in sys.argv:
        return
    texts = {"AMI meetings": [x["text"] for x in inputs("ami", full=True)],
             "FLORES (English)": [x["text"] for x in inputs("flores_eng", full=True)]}
    rows = []
    for n in STEPS:
        v = deepcopy(base)
        v.lemmas = set(base.lemmas) | set(missing[:n])
        for name, sents in texts.items():
            P = [plan(s, v) for s in sents]
            oov = sum(p.n_oov for p in P)
            cont = sum(p.content for p in P)
            rows.append({"added_lemmas": n, "text": name, "sentences": len(sents),
                         "fs_rate": oov / cont, "letters_per_sentence": sum(p.letters_oov for p in P) / len(P),
                         "signing_time_s": sum(p.frames for p in P) / len(P) / FPS})
        print(n, rows[-2]["fs_rate"], rows[-1]["fs_rate"], flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "lexicon_growth.csv", index=False)
    lines = ["| Lemmas added to the IS lexicon | Text | Content words fingerspelled | Letters per sentence | Signing time per sentence, s |",
             "| --- | --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        lines.append(f"| {r['added_lemmas']:,} | {r['text']} | {100 * r['fs_rate']:.1f}% | "
                     f"{r['letters_per_sentence']:.1f} | {r['signing_time_s']:.1f} |")
    (out / "lexicon_growth.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
