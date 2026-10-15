#!/usr/bin/env python3
"""Run the experiments of the vocabulary-constrained translation study (CPU only).

    python code/run.py prepare            # samples, vocabularies, manifest
    python code/run.py nllb               # NLLB translations of FLORES devtest (4 languages)
    python code/run.py swaps              # word-substitution systems (WordNet, embeddings)
    python code/run.py llm [--only JOB]   # EuroLLM systems, resumable job queue

Outputs: results/outputs/<corpus>/<system>.jsonl, one JSON record per sentence.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from sgt import corpora  # noqa: E402
from sgt.vocab import get_vocabulary, STOP  # noqa: E402

# Contractions are allowed like other function words; without them the restricted
# decoder turns "don't" into "done".
CONTRACTIONS = {"don't", "doesn't", "didn't", "can't", "won't", "isn't", "aren't", "wasn't",
                "weren't", "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't", "couldn't",
                "i'm", "you're", "we're", "they're", "it's", "that's", "there's", "here's",
                "what's", "let's", "he's", "she's", "i've", "you've", "we've", "they've", "i'll",
                "you'll", "we'll", "they'll", "it'll", "i'd", "you'd", "we'd", "they'd"}

DATA = ROOT / "data"
OUT = ROOT / "results" / "outputs"
PER_CLIP = DATA / "per_clip_stats.csv"
FLORES_ROOT = DATA / "raw" / "flores200_dataset"
AMI_REF = DATA / "raw" / "ami"
EUROLLM = os.environ.get("EUROLLM_GGUF", str(ROOT / "models" / "EuroLLM-1.7B-Instruct.Q8_0.gguf"))
NLLB_DIR = os.environ.get("NLLB_CT2", str(ROOT / "models" / "nllb-200-distilled-600M-ct2-int8"))
N_SAMPLE = int(os.environ.get("N_SAMPLE", "200"))
N_LLM_FLORES = 100   # FLORES sentences for the LLM systems (CPU budget)
LLM_SEED = 7
SRC_LANGS = ["ell_Grek", "fra_Latn", "spa_Latn", "deu_Latn"]
VOCABS_ALL = ["is", "freq500", "freq1000", "freq2000", "freq5000"]


# ---------------------------------------------------------------- io helpers
def out_path(corpus: str, system: str) -> Path:
    p = OUT / corpus / f"{system}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def done_ids(p: Path) -> set:
    return {r["id"] for r in read_jsonl(p)}


def append(p: Path, rec: dict) -> None:
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def manifest() -> dict:
    return json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def inputs(corpus: str, full: bool = False, llm: bool = False) -> list[dict]:
    """[{'id', 'text', 'ref'?, 'src'?}] for corpus 'flores_eng', 'flores_<lang>' or 'ami'.
    llm=True: the smaller FLORES subset used for the LLM systems (CPU budget)."""
    m = manifest()
    if corpus == "ami":
        items = corpora.load_ami(AMI_REF)
        if full:
            return items
        keep = set(m["ami_ids"])
        return [x for x in items if x["id"] in keep]
    fl = corpora.load_flores(FLORES_ROOT)
    ids = range(len(fl["eng_Latn"])) if full else (m["flores_llm_ids"] if llm else m["flores_ids"])
    lang = "eng_Latn" if corpus == "flores_eng" else corpus.replace("flores_", "")
    return [{"id": i, "text": fl[lang][i], "ref": fl["eng_Latn"][i], "src": fl["ell_Grek"][i]
             if lang == "eng_Latn" else fl[lang][i]} for i in ids]


# ---------------------------------------------------------------- prepare
def cmd_prepare(_):
    fl = corpora.load_flores(FLORES_ROOT)
    n_fl = len(fl["eng_Latn"])
    ami = corpora.load_ami(AMI_REF)
    m = {
        "seed": corpora.SEED,
        "n_sample": N_SAMPLE,
        "flores_devtest_size": n_fl,
        "flores_ids": corpora.sample_ids(n_fl, N_SAMPLE),
        "ami_pool_size": len(ami),
        "ami_ids": [ami[i]["id"] for i in corpora.sample_ids(len(ami), N_SAMPLE)],
        "vocabularies": {},
    }
    # the FLORES subset for the LLM systems: 100 of the sampled sentences (seed 7)
    m["n_llm_flores"] = N_LLM_FLORES
    m["flores_llm_ids"] = sorted(random.Random(LLM_SEED).sample(m["flores_ids"], N_LLM_FLORES))
    for v in VOCABS_ALL:
        voc = get_vocabulary(v, PER_CLIP)
        m["vocabularies"][v] = {"lemmas": len(voc.lemmas), "multiword": len(voc.multiword),
                                "allowed_forms": len(voc.allowed_forms)}
    d = get_vocabulary("is", PER_CLIP).durations
    m["durations_frames"] = {"sign_median": d.sign, "multiword_median": d.multiword,
                             "letter_median": d.letter, "number_median": d.number}
    (DATA / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if not k.endswith("_ids")}, indent=2))


# ---------------------------------------------------------------- NLLB
def cmd_nllb(a):
    from sgt.nllb import NLLB
    tr = NLLB(NLLB_DIR, threads=2)
    for lang in SRC_LANGS:
        items = inputs(f"flores_{lang}", full=a.full)
        p = out_path(f"flores_{lang}", "T0_nllb")
        todo = [x for x in items if x["id"] not in done_ids(p)]
        for i in range(0, len(todo), 32):
            chunk = todo[i:i + 32]
            res = tr.translate([x["text"] for x in chunk], lang)
            for x, (hyp, secs) in zip(chunk, res):
                append(p, {"id": x["id"], "input": x["text"], "output": hyp, "seconds": secs})
        print(lang, "done", len(read_jsonl(p)))


# ---------------------------------------------------------------- substitutions
def cmd_swaps(a):
    from sgt.swaps import EmbeddingIndex, substitute
    vocabs = a.vocabs.split(",") if a.vocabs else VOCABS_ALL
    for corpus in ("flores_eng", "ami"):          # E0: the input itself
        p = out_path(corpus, "E0_orig")
        done = done_ids(p)
        for x in inputs(corpus, full=a.full):
            if x["id"] not in done:
                append(p, {"id": x["id"], "input": x["text"], "output": x["text"], "seconds": 0.0})
    jobs = [("flores_eng", None), ("ami", None)] + [(f"flores_{l}", "T0_nllb") for l in SRC_LANGS]
    for v in vocabs:
        voc = get_vocabulary(v, PER_CLIP)
        idx = EmbeddingIndex(voc)
        for corpus, upstream in jobs:
            if upstream:
                prev = {r["id"]: r for r in read_jsonl(out_path(corpus, upstream))}
                items = [{"id": i, "text": r["output"], "t_up": r["seconds"]} for i, r in prev.items()]
                systems = [("T2_nllb_embed", "embedding")]
            else:
                items = inputs(corpus, full=a.full)
                systems = [("E1_wordnet", "wordnet"), ("E2_embed", "embedding")]
            for sysname, method in systems:
                p = out_path(corpus, f"{sysname}__{v}")
                done = done_ids(p)
                for x in items:
                    if x["id"] in done:
                        continue
                    t0 = time.perf_counter()
                    new, reps = substitute(x["text"], voc, method, idx, tau=a.tau)
                    dt = time.perf_counter() - t0
                    append(p, {"id": x["id"], "input": x["text"], "output": new, "seconds": dt,
                               "upstream_seconds": x.get("t_up", 0.0), "replacements": reps})
                print(v, corpus, sysname, "done")


# ---------------------------------------------------------------- LLM
def llm_jobs():
    """Job queue in priority order: (system, corpus, vocabulary or None).

    Hint prompts with restricted decoding (E5, T3, T5) often made EuroLLM-1.7B copy the
    hint lists in a pilot run, so they are run only for the IS lexicon on English
    FLORES (E5), as an ablation."""
    L2 = ["ell_Grek", "fra_Latn"]
    EN = ["flores_eng", "ami"]
    J = [("T1_eurollm", f"flores_{l}", None) for l in L2]
    J += [("E6_simplify_constrained", c, "is") for c in EN]
    J += [("E5_hints_constrained", "flores_eng", "is")]
    J += [("T6_nllb_simplify_constrained", f"flores_{l}", "is") for l in L2]
    J += [("T4_eurollm_constrained", f"flores_{l}", "is") for l in L2]
    J += [("E3_simplify", c, None) for c in EN]
    J += [("E4_hints", c, "is") for c in EN]
    for v in ["freq1000", "freq5000"]:
        J += [("E6_simplify_constrained", c, v) for c in EN]
        J += [("T6_nllb_simplify_constrained", f"flores_{l}", v) for l in L2]
        J += [("T4_eurollm_constrained", f"flores_{l}", v) for l in L2]
    J += [("E6nofilter_simplify_constrained", "flores_eng", "is")]   # ablation: country names allowed
    for l in ["spa_Latn", "deu_Latn"]:
        J += [("T1_eurollm", f"flores_{l}", None), ("T6_nllb_simplify_constrained", f"flores_{l}", "is")]
    return J


def extras_from(text: str, voc) -> list[str]:
    """Tokens of a sentence the constrained decoder may copy verbatim: its names,
    numbers and signable words (in their original spelling)."""
    from sgt.signing import tokens_with_roles
    out = ["I"]
    for t, role, _ in tokens_with_roles(text, voc):
        if role in ("name", "number", "sign", "sign_fn", "sign_mw", "mw_cont") \
                and t.text.replace(".", "").replace(",", "").replace("'", "").isalnum():
            out.append(t.text)
    return out


def hints_for(text: str, voc, idx) -> list[tuple[str, list[str]]]:
    from sgt.signing import tokens_with_roles
    from sgt.swaps import candidates
    seen, hints = set(), []
    for t, role, _ in tokens_with_roles(text, voc):
        if role == "oov" and t.lower_ not in seen:
            seen.add(t.lower_)
            hints.append((t.text, candidates(t.lower_, t.lemma_.lower(), t.pos_, voc, idx)))
    return hints


def cmd_llm(a):
    from sgt.llm import EuroLLM
    from sgt.signing import plan
    from sgt.swaps import EmbeddingIndex
    model = EuroLLM(EUROLLM, n_threads=a.threads)
    tag = Path(EUROLLM).stem.replace(".", "_")
    vocab_cache, index_cache = {}, {}

    def V(name):
        if name not in vocab_cache:
            vocab_cache[name] = get_vocabulary(name, PER_CLIP)
            index_cache[name] = EmbeddingIndex(vocab_cache[name])
        return vocab_cache[name], index_cache[name]

    for system, corpus, v in llm_jobs():
        name = system if v is None else f"{system}__{v}"
        if a.only and a.only not in name + "@" + corpus:
            continue
        if a.model_tag:
            name = f"{name}@{a.model_tag}"
        p = out_path(corpus, name)
        items = inputs(corpus, llm=True)
        if a.limit:
            items = items[: a.limit]
        done = done_ids(p)
        todo = [x for x in items if x["id"] not in done]
        if not todo:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] {name} on {corpus}: {len(todo)} to do", flush=True)
        voc, idx = V(v) if v else (None, None)
        allowed = (voc.allowed_forms | set(STOP) | CONTRACTIONS) if voc else None
        upstream = {}
        if system in ("T3_nllb_constrained", "T6_nllb_simplify_constrained"):
            upstream = {r["id"]: r for r in read_jsonl(out_path(corpus, "T0_nllb"))}
        if system in ("T4_eurollm_constrained", "T5_eurollm_hints_constrained"):
            t1name = "T1_eurollm" + (f"@{a.model_tag}" if a.model_tag else "")
            upstream = {r["id"]: r for r in read_jsonl(out_path(corpus, t1name))}
        lang = corpus.replace("flores_", "")
        for x in todo:
            rec = {"id": x["id"], "input": x["text"], "model": tag}
            text = x["text"]
            if system == "T1_eurollm":
                g = model.translate(text, lang)
            elif system == "E3_simplify":
                g = model.simplify(text)
            elif system in ("E6_simplify_constrained", "T6_nllb_simplify_constrained",
                            "E6nofilter_simplify_constrained"):
                if system == "T6_nllb_simplify_constrained":
                    up = upstream.get(x["id"])
                    if up is None:
                        continue
                    text = up["output"]
                    rec["upstream_output"] = text
                    rec["upstream_seconds"] = up["seconds"]
                pl = plan(text, voc)
                if pl.n_oov == 0:
                    rec.update(output=text, skipped=True, seconds=0.0)
                    append(p, rec)
                    continue
                if system == "E6nofilter_simplify_constrained":
                    from sgt.vocab import english_forms
                    with_countries = allowed | {f for c in voc.countries for f in
                                                (english_forms().get(c, set()) | {c})}
                    gram = model.grammar(v + "_nofilter", with_countries, extras_from(text, voc))
                else:
                    gram = model.grammar(v, allowed, extras_from(text, voc))
                g = model.simplify(text, gram)
            elif system in ("E4_hints", "E5_hints_constrained", "T3_nllb_constrained"):
                if system == "T3_nllb_constrained":
                    up = upstream.get(x["id"])
                    if up is None:
                        continue
                    text = up["output"]
                    rec["upstream_output"] = text
                    rec["upstream_seconds"] = up["seconds"]
                pl = plan(text, voc)
                if pl.n_oov == 0:
                    rec.update(output=text, skipped=True, seconds=0.0, hints=[])
                    append(p, rec)
                    continue
                hints = hints_for(text, voc, idx)
                gram = None
                if system != "E4_hints":
                    gram = model.grammar(v, allowed, extras_from(text, voc))
                g = model.rewrite_with_hints(text, hints, gram)
                rec["hints"] = hints
            elif system in ("T4_eurollm_constrained", "T5_eurollm_hints_constrained"):
                up = upstream.get(x["id"])
                if up is None:
                    continue
                ex = extras_from(up["output"], voc)
                gram = model.grammar(v, allowed, ex)
                if system == "T4_eurollm_constrained":
                    g = model.translate(text, lang, gram)
                else:
                    hints = hints_for(up["output"], voc, idx)
                    rec["hints"] = hints
                    rec["upstream_output"] = up["output"]
                    g = model.translate_with_hints(text, lang, hints, gram)
            else:
                raise ValueError(system)
            rec.update(output=g.text, raw=g.raw, seconds=g.seconds, prompt_tokens=g.prompt_tokens,
                       completion_tokens=g.completion_tokens, finish=g.finish_reason, skipped=False)
            append(p, rec)
        print(f"[{time.strftime('%H:%M:%S')}] {name} on {corpus}: done", flush=True)


def cmd_sample(a):
    """Best-of-N under the grammar: N sampled restricted simplifications per sentence
    (temperature 0.8, seeds 1..N) are stored next to the greedy output (E6/T6);
    rerank.py then keeps the candidate closest in meaning to the input."""
    from sgt.llm import EuroLLM
    from sgt.signing import plan
    model = EuroLLM(EUROLLM, n_threads=a.threads)
    voc = get_vocabulary(a.vocab, PER_CLIP)
    allowed = voc.allowed_forms | set(STOP) | CONTRACTIONS
    for corpus in a.corpora.split(","):
        greedy_sys = "E6_simplify_constrained" if corpus in ("flores_eng", "ami") else "T6_nllb_simplify_constrained"
        greedy = {r["id"]: r for r in read_jsonl(out_path(corpus, f"{greedy_sys}__{a.vocab}"))}
        p = out_path(corpus, f"S{a.n}_{greedy_sys}__{a.vocab}")
        done = done_ids(p)
        for x in inputs(corpus, llm=True):
            g0 = greedy.get(x["id"])
            if g0 is None or x["id"] in done:
                continue
            text = g0.get("upstream_output", x["text"])
            rec = {"id": x["id"], "input": x["text"], "rewrite_input": text,
                   "upstream_seconds": g0.get("upstream_seconds", 0.0)}
            if g0.get("skipped"):
                rec.update(candidates=[text], seconds=[0.0], skipped=True)
                append(p, rec)
                continue
            cands, secs = [g0["output"]], [g0["seconds"]]
            for k in range(1, a.n + 1):
                g = model.simplify(text, model.grammar(a.vocab, allowed, extras_from(text, voc)),
                                   temperature=0.8, seed=k)
                cands.append(g.text)
                secs.append(g.seconds)
            rec.update(candidates=cands, seconds=secs, skipped=False)
            append(p, rec)
        print(corpus, "sampled", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    s = sub.add_parser("nllb"); s.add_argument("--full", action="store_true")
    s = sub.add_parser("swaps"); s.add_argument("--full", action="store_true")
    s.add_argument("--vocabs", default=""); s.add_argument("--tau", type=float, default=0.5)
    s = sub.add_parser("llm"); s.add_argument("--only", default="")
    s.add_argument("--threads", type=int, default=2); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--model-tag", default="")
    s = sub.add_parser("sample"); s.add_argument("--vocab", default="is")
    s.add_argument("--corpora", default="flores_eng"); s.add_argument("--n", type=int, default=4)
    s.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "nllb": cmd_nllb, "swaps": cmd_swaps, "llm": cmd_llm,
     "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
