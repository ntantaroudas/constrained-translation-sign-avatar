#!/usr/bin/env python3
"""Error analysis of restricted decoding.

    python code/errors.py      # writes results/tables/errors_*.md and results/errors/*.csv

1. Country and territory glosses. The IS glosses that name a country or a
   territory are found by fuzzy matching (difflib ratio >= 0.85) against the
   ISO 3166-1 names in pycountry, which also catches misspelled glosses. They
   are compared with the WordNet test used to keep country names out of the
   restricted vocabulary (sgt.vocab.is_country), and every restricted output is
   searched for a country or territory gloss that its input does not mention.

2. Prefix substitutions. For English input, a word that the input would
   fingerspell (an avoidable fingerspelling, from the signing plan of the input)
   counts as replaced by a prefix substitute when the output contains a word
   that is not in the input, starts with the same three letters, and is not a
   derivational relative of it (same Porter stem, or one stem a prefix of the
   other). Example: "Parliament" -> "Paraguay". The rate is reported per
   avoidable fingerspelling in the input, for the restricted systems and, as a
   control for chance overlaps, for the unrestricted and word-level systems.

3. Dropped stop-list content words. The restricted grammar allows every word of the
   scikit-learn stop list, which also contains content words (give, show, part,
   system). The signing model drops stop words that have no entry, so these words are
   neither signed nor fingerspelled. Counted: tokens whose form or lemma is on the stop
   list, tagged as noun, verb or adjective and dropped, in the outputs and in the texts
   they rewrite.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from run import OUT, PER_CLIP, manifest, read_jsonl  # noqa: E402
from sgt.vocab import get_vocabulary, lemma  # noqa: E402

TABLES = ROOT / "results" / "tables"
ERR = ROOT / "results" / "errors"
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def plain(text: str) -> str:
    """Lower case without accents: 'Côte d'Ivoire' -> 'cote d'ivoire'."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def country_names() -> list[str]:
    import pycountry
    names = set()
    for c in pycountry.countries:
        for attr in ("name", "common_name", "official_name"):
            n = getattr(c, attr, None)
            if n:
                names.add(n.lower())
                names.add(re.sub(r"\s*\(.*\)", "", n).lower())  # "Holy See (Vatican City State)"
                if "," in n:  # "Korea, Republic of" -> both orders
                    a, b = [x.strip() for x in n.split(",", 1)]
                    names.add(f"{b} {a}".lower())
    names |= {"south korea", "north korea", "democratic republic of congo", "republic of congo",
              "vatican", "ivory coast", "east timor", "swaziland", "burma", "macedonia", "czech republic",
              "kosovo", "palestine", "taiwan", "russia", "syria", "iran", "laos", "vietnam", "bolivia",
              "venezuela", "tanzania", "moldova", "brunei", "micronesia", "sahrawi arab democratic republic",
              "wales", "korea"}  # a constituent country and a short name not in ISO 3166-1
    return sorted({plain(n) for n in names})


def is_country_gloss(gloss: str, names: list[str]) -> str | None:
    """Exact match; multiword glosses may also match approximately, which catches
    misspelled glosses such as 'central african rebublic' (single words are matched
    exactly only, since 'man' or 'hungry' are close to 'Oman' or 'Hungary')."""
    g = plain(gloss)
    if g in names:
        return g
    if " " not in g:
        return None
    best = difflib.get_close_matches(g, names, n=1, cutoff=0.85)
    return best[0] if best else None


def gloss_list(v) -> list[str]:
    single = sorted({x for x in v.lemmas})
    multi = sorted(" ".join(m) if not isinstance(m, str) else m for m in v.multiword)
    return single + multi


def contains_phrase(text: str, phrase: str) -> bool:
    return re.search(r"(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])", text, flags=re.I) is not None


def country_analysis(v) -> tuple[pd.DataFrame, list[str]]:
    names = country_names()
    rows = []
    for g in gloss_list(v):
        if re.fullmatch(r"[a-z]", g) or g.isdigit():
            continue
        match = is_country_gloss(g, names)
        if match:
            rows.append({"gloss": g, "iso_name": match, "multiword": " " in g,
                         "wordnet_country": g in v.countries})
    matched = {r["gloss"] for r in rows}
    for g in sorted(set(v.countries) - matched):  # e.g. england, scotland: not ISO 3166-1 countries
        rows.append({"gloss": g, "iso_name": "", "multiword": " " in g, "wordnet_country": True})
    df = pd.DataFrame(rows)
    return df, sorted(df["gloss"])


REFERENCE_SYSTEM = {"T2": "T0_nllb", "T6": "T0_nllb", "T3": "T0_nllb", "T4": "T1_eurollm", "T5": "T1_eurollm"}


def rewritten_text(corpus: str, system: str):
    """id -> the English text a system rewrote: the input for English input, the NLLB
    translation for the NLLB cascades and EuroLLM's own translation for T4."""
    ref = REFERENCE_SYSTEM.get(system[:2])
    if ref is None:
        return lambda r: r["input"]
    up = {r["id"]: r["output"] for r in read_jsonl(OUT / corpus / f"{ref}.jsonl")}
    return lambda r: r.get("upstream_output") or up[r["id"]]


def leak_analysis(country_glosses: list[str]) -> pd.DataFrame:
    ids = set(manifest()["flores_llm_ids"])
    rows = []
    for corpus in ["flores_eng", "ami", "flores_ell_Grek", "flores_fra_Latn", "flores_spa_Latn",
                   "flores_deu_Latn"]:
        for f in sorted((OUT / corpus).glob("*__is.jsonl")):
            if re.match(r"S\d+_", f.stem) or "constrained" not in f.stem:
                continue
            recs = [r for r in read_jsonl(f) if corpus == "ami" or r["id"] in ids]
            src_of = rewritten_text(corpus, f.stem)
            n_leak, examples = 0, []
            for r in recs:
                src = src_of(r)
                leaked = [g for g in country_glosses
                          if contains_phrase(r["output"], g) and not contains_phrase(src, g)
                          and not contains_phrase(r["input"], g)]
                if leaked:
                    n_leak += 1
                    examples.append(f"{r['id']}: {', '.join(leaked)}")
            rows.append({"corpus": corpus, "system": f.stem, "n": len(recs), "sentences_with_leak": n_leak,
                         "share": n_leak / max(len(recs), 1), "examples": "; ".join(examples[:6])})
    return pd.DataFrame(rows)


def prefix_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    from nltk.stem import PorterStemmer
    stem = PorterStemmer().stem
    sent = pd.read_csv(ROOT / "results" / "scores" / "sentences.csv", usecols=[
        "corpus", "system", "id", "vocab", "oov_words"])
    m = manifest()
    llm_ids = set(m["flores_llm_ids"])
    oov = {}  # (corpus, reference system, vocab, id) -> avoidable fingerspellings of the rewritten text
    ref_rows = sent[sent.system.isin(["E0_orig", "T0_nllb", "T1_eurollm"])]
    for r in ref_rows.itertuples():
        oov[(r.corpus, r.system, r.vocab, str(r.id))] = \
            str(r.oov_words).split() if isinstance(r.oov_words, str) else []

    def related(a: str, b: str) -> bool:
        sa, sb = stem(a), stem(b)
        return sa == sb or sa.startswith(sb) or sb.startswith(sa) or lemma(a) == lemma(b)

    rows, pairs = [], []
    for corpus in ["flores_eng", "ami", "flores_ell_Grek", "flores_fra_Latn", "flores_spa_Latn",
                   "flores_deu_Latn"]:
        for f in sorted((OUT / corpus).glob("*.jsonl")):
            sysn = f.stem
            if re.match(r"S\d+_", sysn) or sysn in ("E0_orig", "T0_nllb", "T1_eurollm"):
                continue
            ref_sys = REFERENCE_SYSTEM.get(sysn[:2], "E0_orig")
            src_of = rewritten_text(corpus, sysn)
            vocabs = [sysn.split("__")[1]] if "__" in sysn else ["is", "freq1000", "freq5000"]
            recs = [r for r in read_jsonl(f) if corpus == "ami" or r["id"] in llm_ids]
            for v in vocabs:
                n_oov = n_trap = n_sent = 0
                for r in recs:
                    words_in = {w.lower() for w in WORD.findall(src_of(r))}
                    lemmas_in = {lemma(w) for w in words_in}
                    out_words = {w.lower() for w in WORD.findall(r["output"])}
                    new = [w for w in out_words if len(w) >= 3 and w not in words_in
                           and lemma(w) not in lemmas_in]
                    u_list = [u.lower() for u in oov.get((corpus, ref_sys, v, str(r["id"])), [])]
                    hit = False
                    for u in u_list:
                        n_oov += 1
                        subs = [w for w in new if len(u) >= 3 and w[:3] == u[:3] and not related(w, u)]
                        if subs:
                            n_trap += 1
                            hit = True
                            pairs.append({"corpus": corpus, "system": sysn, "vocab": v, "id": r["id"],
                                          "input_word": u, "output_word": subs[0]})
                    n_sent += hit
                rows.append({"corpus": corpus, "system": sysn, "vocab": v, "sentences": len(recs),
                             "avoidable_fingerspellings_in_input": n_oov, "prefix_substituted": n_trap,
                             "rate": n_trap / max(n_oov, 1), "sentences_with_one": n_sent / max(len(recs), 1)})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def dropped_stop_words() -> pd.DataFrame:
    from collections import Counter
    from sgt.signing import tokens_with_roles
    from sgt.vocab import STOP
    v = get_vocabulary("is", PER_CLIP)
    ids = set(manifest()["flores_llm_ids"])

    def count(text: str) -> Counter:
        c = Counter()
        for t, role, _ in tokens_with_roles(text, v):
            # the signing model's own rule: surface form or lemma on the stop list
            if role == "drop" and (t.lower_ in STOP or t.lemma_.lower() in STOP) \
                    and t.pos_ in {"NOUN", "VERB", "ADJ"}:
                c[t.lower_] += 1
        return c
    rows = []
    for corpus, system in [("flores_eng", "E3_simplify"), ("flores_eng", "E6_simplify_constrained__is"),
                           ("flores_eng", "R5_E6_simplify_constrained__is"), ("ami", "E3_simplify"),
                           ("ami", "E6_simplify_constrained__is"), ("flores_ell_Grek", "T6_nllb_simplify_constrained__is"),
                           ("flores_ell_Grek", "T4_eurollm_constrained__is"), ("flores_fra_Latn", "T6_nllb_simplify_constrained__is"),
                           ("flores_fra_Latn", "T4_eurollm_constrained__is")]:
        src_of = rewritten_text(corpus, system)
        recs = [r for r in read_jsonl(OUT / corpus / f"{system}.jsonl") if corpus == "ami" or r["id"] in ids]
        ci, co = Counter(), Counter()
        for r in recs:
            ci += count(src_of(r))
            co += count(r["output"])
        rows.append({"corpus": corpus, "system": system, "sentences": len(recs), "in_rewritten_text": sum(ci.values()),
                     "in_output": sum(co.values()), "most_common_in_output": " ".join(f"{w}:{n}" for w, n in co.most_common(10))})
    return pd.DataFrame(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    ERR.mkdir(parents=True, exist_ok=True)
    v = get_vocabulary("is", PER_CLIP)
    cdf, glosses = country_analysis(v)
    cdf.to_csv(ERR / "country_glosses.csv", index=False)
    missed = cdf[~cdf.wordnet_country]
    wn_only = sorted(cdf[cdf.iso_name == ""].gloss)
    leak = leak_analysis(glosses)
    leak.to_csv(ERR / "country_leaks.csv", index=False)
    drops = dropped_stop_words()
    drops.to_csv(ERR / "dropped_stop_words.csv", index=False)
    pre, pairs = prefix_analysis()
    pre.to_csv(ERR / "prefix_substitution.csv", index=False)
    pairs.to_csv(ERR / "prefix_pairs.csv", index=False)

    lines = ["# Country and territory glosses", "",
             f"IS glosses naming a country or territory: {len(cdf)} "
             f"({int((~cdf.multiword).sum())} single-word, {int(cdf.multiword.sum())} multiword); "
             f"matching an ISO 3166-1 name: {int((cdf.iso_name != '').sum())}.",
             f"Of these, recognised by the WordNet test: {int(cdf.wordnet_country.sum())}; missed: {len(missed)}.",
             "Missed: " + ", ".join(f"{g} ({n})" for g, n in zip(missed.gloss, missed.iso_name)),
             f"Flagged by WordNet but not matched to an ISO name: {', '.join(wn_only) or 'none'}", "",
             "# Restricted outputs naming a country or territory that the input does not mention", "",
             "| Text | System | Sentences | With such a name | Share, % | Examples |", "| --- | --- | --- | --- | --- | --- |"]
    for r in leak.itertuples():
        lines.append(f"| {r.corpus} | {r.system} | {r.n} | {r.sentences_with_leak} | {100 * r.share:.1f} | {r.examples} |")
    lines += ["", "# Avoidable fingerspellings replaced by an unrelated word with the same first three letters", "",
              "| Text | System | Vocabulary | Avoidable fingerspellings in input | Replaced by a prefix substitute | Rate, % | Sentences with at least one, % |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in pre.sort_values(["corpus", "vocab", "system"]).itertuples():
        lines.append(f"| {r.corpus} | {r.system} | {r.vocab} | {r.avoidable_fingerspellings_in_input} | "
                     f"{r.prefix_substituted} | {100 * r.rate:.1f} | {100 * r.sentences_with_one:.1f} |")
    lines += ["", "# Stop-list content words dropped by the signing model (IS lexicon)", "",
              "| Text | System | Sentences | In the rewritten text | In the output | Most common in the output |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in drops.itertuples():
        lines.append(f"| {r.corpus} | {r.system} | {r.sentences} | {r.in_rewritten_text} | {r.in_output} | "
                     f"{r.most_common_in_output} |")
    (TABLES / "errors.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
