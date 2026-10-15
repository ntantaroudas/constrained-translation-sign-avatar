#!/usr/bin/env python3
"""Tables of the manuscript, generated from the data and result files (no number is typed by hand,
except the settings of Table 3, which describe how the experiments were run).

    python code/paper_tables.py     # -> manuscript/src/06_tables.md, or results/tables/article_tables.md
                                    #    when the manuscript sources are not present

Reads data/per_clip_stats.csv, data/manifest.json, results/tables/main_results.csv,
results/tables/lexicon_growth.csv, results/tables/nllb_latency.csv, results/scores/sentences.csv,
results/errors/*.csv and the system outputs. The text of the article explains the results in words
and refers to these tables for the numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from analysis import LABEL, VOCAB_LABEL, boot_ci, paired_diff, ratio_ci  # noqa: E402,F401

TABLES = ROOT / "results" / "tables"
SCORES = ROOT / "results" / "scores"
ERR = ROOT / "results" / "errors"
OUT = ROOT / "results" / "outputs"
SRC = ROOT / "manuscript" / "src"
FPS = 30.0

# Paper code of each result file stem (the files keep the names of the experiment run, T6 = paper T3).
CODE = {"E0_orig": "E0", "E1_wordnet": "E1", "E2_embed": "E2", "E3_simplify": "E3", "E4_hints": "E4",
        "E5_hints_constrained": "E5", "E6_simplify_constrained": "E6",
        "E6nofilter_simplify_constrained": "E6c", "R5_E6_simplify_constrained": "E6r",
        "T0_nllb": "T0", "T1_eurollm": "T1", "T2_nllb_embed": "T2", "T6_nllb_simplify_constrained": "T3",
        "T4_eurollm_constrained": "T4"}
# Plain-language names of the methods, as in Table 1.
NAME = {"E0_orig": "Input as is", "E1_wordnet": "WordNet replacement", "E2_embed": "Embedding replacement",
        "E3_simplify": "Simplification by the language model", "E4_hints": "Rewrite with suggested words",
        "E5_hints_constrained": "Rewrite with suggested words, restricted",
        "E6_simplify_constrained": "Simplification, restricted",
        "E6nofilter_simplify_constrained": "Simplification, restricted, all country names allowed",
        "R5_E6_simplify_constrained": "Simplification, restricted, best of five",
        "T0_nllb": "NLLB translation", "T1_eurollm": "EuroLLM translation",
        "T2_nllb_embed": "NLLB, then embedding replacement",
        "T6_nllb_simplify_constrained": "NLLB, then restricted simplification",
        "T4_eurollm_constrained": "EuroLLM, restricted translation"}
LEX = {"is": "IS", "freq500": "500", "freq1000": "1,000", "freq2000": "2,000", "freq5000": "5,000"}
DICT = {"is": "IS dictionary", "freq500": "Top 500 words", "freq1000": "Top 1,000 words",
        "freq2000": "Top 2,000 words", "freq5000": "Top 5,000 words"}
TEXT = {"flores_eng": "Written, English", "ami": "Meetings", "flores_ell_Grek": "Written, Greek",
        "flores_fra_Latn": "Written, French", "flores_spa_Latn": "Written, Spanish", "flores_deu_Latn": "Written, German"}
DASH = "–"


def base(system: str) -> str:
    return system.split("__")[0]


def f1(x, nd=1, k=1.0):
    return DASH if x is None or (isinstance(x, float) and np.isnan(x)) else f"{k * x:.{nd}f}"


def ci(x, c, nd=1, k=1.0):
    if isinstance(c, str):
        c = tuple(float(v) for v in c.strip("()").split(","))
    return f"{k * x:.{nd}f} ({k * c[0]:.{nd}f}–{k * c[1]:.{nd}f})"


def ci_or_value(x, c, nd=1, k=1.0):
    """Value with its interval, or the value alone when the interval has no width (the unchanged input)."""
    if isinstance(c, str):
        c = tuple(float(v) for v in c.strip("()").split(","))
    if round(k * c[0], nd) == round(k * c[1], nd):
        return f"{k * x:.{nd}f}"
    return ci(x, c, nd=nd, k=k)


def signed(x: float, nd: int) -> str:
    t = f"{x:+.{nd}f}"
    t = t[1:] if float(t) == 0 else t
    return t.replace("-", "−")


def res() -> pd.DataFrame:
    return pd.read_csv(TABLES / "main_results.csv")


def row(r: pd.DataFrame, corpus: str, vocab: str, system: str):
    name = system if system in ("E0_orig", "E3_simplify", "T0_nllb", "T1_eurollm") else f"{system}__{vocab}"
    x = r[(r.corpus == corpus) & (r.vocab == vocab) & (r.system == name)]
    return None if x.empty else x.iloc[0]


def md_table(caption: str, header: list[str], rows: list[list[str]], note: str) -> str:
    # soft hyphens let long header words break cleanly in narrow columns of the Word file
    header = [h.replace("fingerspell", "finger\u00adspell").replace("Fingerspell", "Finger\u00adspell") for h in header]
    lines = [f"@TABLE {caption}", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines.append(f"@NOTE {note}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- Table 1
def table1() -> str:
    rows = [
        ["E0", "English", "The sentence is signed as it is (baseline)", "All five", "200", "200"],
        ["E1", "English", "Each content word without a sign is replaced by a WordNet synonym or more general word that has one", "All five", "200", "200"],
        ["E2", "English", "Each content word without a sign is replaced by the most similar word with a sign, judged by word vectors", "All five", "200", "200"],
        ["E3", "English", "The language model simplifies the sentence in common words", "Not used", "100", "200"],
        ["E4", "English", "The language model rewrites the sentence and is given suggested words that have a sign", "IS", "100", "200"],
        ["E5", "English", "As E4, with the model restricted to words the avatar can sign", "IS", "100", DASH],
        ["E6", "English", "As E3, with the model restricted to words the avatar can sign", "IS, 1,000 and 5,000", "100", "200"],
        ["E6c", "English", "As E6, with every country name of the dictionary allowed", "IS", "100", DASH],
        ["E6r", "English", "Five restricted rewrites (E6 and four samples), of which the one closest in meaning to the input is kept", "IS", "100", DASH],
        ["T0", "Greek, French, Spanish, German", "Translation into English by NLLB", "Not used", "200", DASH],
        ["T1", "Greek, French, Spanish, German", "Translation into English by EuroLLM", "Not used", "100", DASH],
        ["T2", "Greek, French, Spanish, German", "NLLB translation, then embedding replacement as in E2", "All five", "200", DASH],
        ["T3", "Greek, French, Spanish, German", "NLLB translation, then restricted simplification as in E6", "IS, 1,000 and 5,000 (Greek, French), IS (Spanish, German)", "100", DASH],
        ["T4", "Greek, French", "EuroLLM translates directly, restricted to words the avatar can sign", "IS, 1,000 and 5,000", "100", DASH],
    ]
    return md_table("Table 1. Methods compared.",
                    ["Code", "Input", "What the method does", "Dictionaries", "Written sentences", "Meeting segments"], rows,
                    "Dictionaries are the International Sign (IS) dictionary and the lists of the 500, 1,000, 2,000 and "
                    "5,000 most frequent English content words. Methods that do not use a dictionary are scored with every "
                    "dictionary. Language model, EuroLLM-1.7B-Instruct. All comparisons between methods on the written "
                    "sentences use the 100 sentences on which every method was run.")


# ----------------------------------------------------------------------------------- Table 2
def table2() -> str:
    from run import PER_CLIP
    from sgt.vocab import is_vocabulary, lemma, ranked_content_lemmas
    clips = pd.read_csv(ROOT / "data" / "per_clip_stats.csv")
    kind = clips.category.value_counts()
    hands = clips[clips.any_hand_frames > 0]
    singles = hands[hands.category == "single_word"].gloss.str.strip().str.lower()
    phrases = hands[hands.category == "multiword"].gloss.str.strip().str.lower()
    vocab = is_vocabulary(PER_CLIP)
    ranked = [w for w, _ in ranked_content_lemmas()]
    cover = {k: len(set(ranked[:k]) & vocab.lemmas) for k in (1000, 5000)}
    dur = json.loads((ROOT / "data" / "manifest.json").read_text())["durations_frames"]
    c = pd.read_csv(ERR / "country_glosses.csv")
    missed = c[~c.wordnet_country]
    misspelled = missed[missed.gloss != missed.iso_name]
    admissible = missed[missed.gloss.map(lambda g: all(t.isalnum() for t in g.split()))]

    def split(d):
        return f"{len(d)} ({int((~d.multiword).sum())} single words, {int(d.multiword.sum())} phrases)"

    rows = [
        ["Clips in the IS dictionary", f"{len(clips)}"],
        ["of which single words, phrases, letters and numerals",
         f"{kind['single_word']}, {kind['multiword']}, {kind['single_letter']} and {kind['number']}"],
        ["Clips left out because no hand was detected", f"{len(clips) - len(hands)}"],
        ["Distinct single-word signs (base forms)", f"{singles.nunique()} ({len({lemma(g) for g in singles})})"],
        ["Phrase signs", f"{phrases.nunique()}"],
        ["The 1,000 most frequent English content words that have a sign",
         f"{cover[1000]} ({100 * cover[1000] / 1000:.1f}%)"],
        ["The 5,000 most frequent English content words that have a sign",
         f"{cover[5000]} ({100 * cover[5000] / 5000:.1f}%)"],
        ["Glosses that name a country or territory", split(c)],
        ["of which recognised by the WordNet test", split(c[c.wordnet_country])],
        ["of which missed by the WordNet test", split(missed)],
        ["of the missed, misspelled in the dictionary", ", ".join(misspelled.gloss)],
        ["of the missed, still allowed in restricted rewriting", f"{len(admissible)}"],
        ["Typical (median) length of a single-word sign, phrase sign, letter and numeral, frames",
         f"{dur['sign_median']:.0f}, {dur['multiword_median']:.0f}, {dur['letter_median']:.0f} and {dur['number_median']:.0f}"],
    ]
    return md_table("Table 2. The dictionaries of signs.", ["Item", "Value"], rows,
                    "The IS dictionary is that of the avatar described in [@tantaroudas2026ai], whose hand-tracking "
                    "files are openly available [@tantaroudas_zenodo]. Only the glosses and the clip lengths were used. "
                    "A gloss is the word or phrase that a sign stands for. The frequency lists contain the 500, 1,000, "
                    "2,000 or 5,000 most frequent English content words, counted by base form, and have no clips, so "
                    "each of their signs is given the typical length of a single-word sign. Frame counts were converted "
                    "to seconds at 30 frames per second, the frame rate of five source clips that were checked, and at these "
                    "lengths a word of four or more letters takes longer to fingerspell than a typical sign. A country "
                    "gloss was missed by the WordNet test when none of its WordNet meanings is an instance of a country "
                    "or a state, and two of the missed glosses contain an apostrophe or a hyphen, which the list of "
                    "allowed words excludes.")


# ----------------------------------------------------------------------------------- Table 3
def table3() -> str:
    rows = [
        ["Meeting speech", "AMI Meeting Corpus, manual transcripts (annotations version 1.6.2) of the 16 evaluation "
         "meetings, with hesitations removed and immediate repetitions collapsed, 2,162 speaker segments of 6 to 40 "
         "words with at least three content words, of which 200 were sampled (seed 42)"],
        ["Written text", "FLORES-200 devtest, 1,012 sentences from Wikinews, Wikivoyage and Wikibooks, available "
         "in English, Greek, French, Spanish and German, of which 200 were sampled (seed 42), and a random 100 of "
         "these (seed 7) for every method that uses a language model"],
        ["Tagging", "spaCy 3.8 with the en_core_web_sm 3.8.0 model"],
        ["Frequency lists", "wordfreq 3.1.1 (60,000 most frequent entries), base forms by simplemma 2.0.0, "
         "scikit-learn English stop words removed, frequencies summed per base form"],
        ["WordNet replacement (E1)", "WordNet 3.0, first dictionary word among the synonyms of the word's meanings, "
         "then among more general words up to two levels, then among similar adjectives"],
        ["Embedding replacement (E2, T2)", "spaCy en_core_web_lg 3.8.0 word vectors, cosine similarity of at least "
         "0.5, same part of speech in WordNet, grammatical form copied with lemminflect 0.2.3"],
        ["Suggested words (E4, E5)", "up to four per word without a sign, up to two from WordNet and the rest from "
         "word vectors with a similarity of at least 0.45"],
        ["Language model", "EuroLLM-1.7B-Instruct in 8-bit GGUF form (Q8_0), run with llama.cpp through "
         "llama-cpp-python 0.3.16 with its chat template, greedy decoding, repetition penalty 1.1, output limited to "
         "three tokens per input word plus 16, and to at least 48 tokens"],
        ["Best of five (E6r)", "the greedy output and four samples (temperature 0.8, top-k 40, top-p 0.95, seeds 1 to "
         "4), of which the one whose all-mpnet-base-v2 sentence embedding is closest to that of the input is kept"],
        ["Translation model", "NLLB-200-distilled-600M with CTranslate2 4.8.2 in 8-bit integer form, beam size 4, "
         "batches of 16 sentences, and EuroLLM with the translation prompt of its model card"],
        ["Restricting the model", "a GBNF grammar for llama.cpp that allows a single sentence of allowed words, "
         "including the stop words and 40 contractions, with more than 12,000 word forms for the 5,000-word list, "
         "stored as a tree of letters"],
        ["Country names", "WordNet 3.0 test for the restriction, and for the failure analysis also the ISO 3166-1 "
         "names of pycountry 26.2.16 with common short names and a few other names such as Wales and Kosovo added, "
         "accents removed and approximate matching of phrases (difflib similarity of at least 0.85)"],
        ["Measures", "COMET with the wmt22-comet-da model, chrF from sacreBLEU 2.5.1, sentence embeddings from "
         "all-mpnet-base-v2, mutual entailment from cross-encoder/nli-deberta-v3-base"],
        ["Computing", "two CPU threads of a cloud container with 8 GB of memory and no GPU"],
        ["Statistics", "95% percentile bootstrap intervals over sentences, with 1,000 resamples (seed 42), paired by "
         "sentence when two methods are compared"],
    ]
    return md_table("Table 3. Texts, models and settings.", ["Item", "Setting"], rows,
                    "Two runs whose timing was disturbed by another process running at the same time, E5 and the "
                    "EuroLLM translation of the Greek sentences, were repeated alone. Their outputs were identical and "
                    "their times were replaced.")


# ----------------------------------------------------------------------------------- Table 4
def table4() -> str:
    rows = [
        ["Content words fingerspelled, %", "Share of the content words that have no sign and are spelled letter by "
         "letter, pooled over sentences. Names without a sign are fingerspelled too but counted separately",
         "Lower is better"],
        ["Signing time, s", "Estimated time to sign a sentence, from the clip lengths, without transitions between "
         "signs", "Shorter is better for the viewer, if the meaning is kept"],
        ["COMET", "How well an output keeps the meaning of a reference sentence, scored by a neural model trained on "
         "human judgements of translations. For English input the reference is the input itself", "Higher is better, "
         "the unchanged English input scores about 0.95"],
        ["chrF", "Score from 0 to 100 for the short letter sequences that the output shares with the reference",
         "Higher means closer wording"],
        ["Cosine similarity", "Closeness in meaning of a meeting segment and its rewrite, judged by a "
         "sentence-similarity model, at most 1",
         "Higher means closer meaning"],
        ["Mutual entailment, %", "Share of meeting segments for which an inference model judges that the rewrite "
         "follows from the original and the original from the rewrite", "Higher is better"],
        ["Time per sentence, s", "Mean computing time per sentence on two CPU threads", "Lower is faster"],
        ["Look-alike words, %", "Share of the words that would have been fingerspelled which were replaced by an "
         "unrelated word that begins with the same three letters (dinosaur became dinner)", "Lower is better, and "
         "methods without restriction show how often this happens by chance"],
        ["Unwanted country names, %", "Share of outputs that name a country or territory which the text being "
         "rewritten does not mention", "Lower is better"],
        ["Dropped stop-list words", "Nouns, verbs and adjectives that the avatar drops because they are on the stop "
         "list, counted in the output and in the text being rewritten", "Closer to the text being rewritten is "
         "better"],
    ]
    return md_table("Table 4. How the results were measured.", ["Measure", "What it shows", "How to read it"], rows,
                    "Where the tables give a 95% confidence interval, it comes from resampling the sentences many "
                    "times (bootstrap). When two methods are compared on the same sentences, a "
                    "confidence interval of their difference that excludes zero indicates a difference beyond chance.")


# ----------------------------------------------------------------------------------- Table 5
def table5() -> str:
    df = pd.read_csv(SCORES / "sentences.csv")
    rows = []
    for corpus, label in [("ami", "Meetings"), ("flores_eng", "Written, English")]:
        for v in ["is", "freq500", "freq1000", "freq2000", "freq5000"]:
            g = df[(df.corpus == corpus) & (df.system == "E0_orig") & (df.vocab == v)].sort_values("id")
            fs = g.n_oov.sum() / g.content.sum()
            fs_ci = ratio_ci(g.n_oov.to_numpy(), g.content.to_numpy())
            t = g.frames.mean() / FPS
            t_ci = tuple(c / FPS for c in boot_ci(g.frames.to_numpy()))
            share = g.fs_oov_frames.sum() / g.frames.sum()
            rows.append([label, DICT[v], ci(fs, fs_ci, k=100), f"{g.letters_oov.mean():.1f}",
                         f"{g.letters_name.mean():.1f}", ci(t, t_ci), f"{100 * share:.1f}"])
    return md_table("Table 5. Fingerspelling and signing time when the text is signed as it is.",
                    ["Text", "Dictionary", "Content words fingerspelled, % (95% CI)",
                     "Fingerspelled letters per sentence, content words", "Fingerspelled letters per sentence, names",
                     "Signing time per sentence, s (95% CI)", "Share of signing time spent fingerspelling content words, %"],
                    rows,
                    "200 meeting segments and 200 FLORES-200 sentences. Content words leave out numbers, function "
                    "words, stop words, pronouns and interjections, also when the dictionary has a sign for them. "
                    "Names without a sign are fingerspelled and counted separately, whereas names that have a sign, "
                    "such as many country names, are signed and counted as content words. The signing time includes "
                    "the fingerspelling of names.")


# ----------------------------------------------------------------------------------- Table 6
def table6() -> str:
    g = pd.read_csv(TABLES / "lexicon_growth.csv")
    rows = []
    for n in sorted(g.added_lemmas.unique()):
        a = g[(g.added_lemmas == n) & (g.text == "AMI meetings")].iloc[0]
        f = g[(g.added_lemmas == n) & (g.text == "FLORES (English)")].iloc[0]
        rows.append([f"{n:,}", f"{100 * a.fs_rate:.1f}", f"{a.signing_time_s:.1f}", f"{100 * f.fs_rate:.1f}",
                     f"{f.signing_time_s:.1f}"])
    yeah = _yeah_counts()
    return md_table("Table 6. Fingerspelling and signing time when the IS dictionary is extended with frequent words.",
                    ["Words added", "Meetings, content words fingerspelled, %", "Meetings, signing time, s",
                     "Written, content words fingerspelled, %", "Written, signing time, s"], rows,
                    "The IS dictionary is extended with the most frequent English content words it lacks, in order of "
                    "frequency and without country names, and the text is signed as it is. All 2,162 meeting segments "
                    "and all 1,012 FLORES-200 sentences. Each added word is given the typical length of a single-word "
                    f"sign. Some added words were dropped before they had a sign. For example, yeah occurs {yeah[0]:,} "
                    f"times in the meeting segments, and {yeah[1]:,} of these occurrences are dropped while it has no "
                    "sign, almost all as interjections.")


def _heading_copies() -> dict:
    """Outputs of E4 and E5 (IS dictionary) with the heading of the list of suggested words, Words that cannot be
    signed. Only the first line of an answer is kept, so an E4 answer that starts with the heading becomes the heading
    alone. Keys, (corpus, system, "only") for outputs that are the heading alone, (corpus, system, "any") for outputs
    that contain it."""
    import re
    out = {}
    for corpus, s in [("flores_eng", "E4_hints__is"), ("ami", "E4_hints__is"), ("flores_eng", "E5_hints_constrained__is")]:
        outs = [json.loads(l)["output"] for l in (OUT / corpus / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
        out[(corpus, s, "only")] = sum(bool(re.fullmatch(r"\s*words that cannot be signed\s*:?\s*", o, re.I)) for o in outs)
        out[(corpus, s, "any")] = sum(bool(re.search(r"cannot be signed", o, re.I)) for o in outs)
    return out


def _yeah_counts() -> tuple[int, int]:
    """Occurrences of 'yeah' in all meeting segments, and how many the signing model drops with the IS lexicon."""
    cache = TABLES / "yeah_counts.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["occurrences"], d["dropped"]
    from run import PER_CLIP, inputs
    from sgt.signing import tokens_with_roles
    from sgt.vocab import get_vocabulary
    vocab = get_vocabulary("is", PER_CLIP)
    occ = dropped = 0
    for item in inputs("ami", full=True):
        for tok, role, _ in tokens_with_roles(item["text"], vocab):
            if tok.lower_ == "yeah":
                occ += 1
                dropped += role == "drop"
    cache.write_text(json.dumps({"occurrences": occ, "dropped": dropped}))
    return occ, dropped


# ----------------------------------------------------------------------------------- Tables 7 and 8
ENGLISH_ORDER = {"is": ["E0_orig", "E1_wordnet", "E2_embed", "E3_simplify", "E4_hints", "E6_simplify_constrained",
                        "E5_hints_constrained", "E6nofilter_simplify_constrained", "R5_E6_simplify_constrained"],
                 "freq1000": ["E0_orig", "E1_wordnet", "E2_embed", "E3_simplify", "E6_simplify_constrained"],
                 "freq5000": ["E0_orig", "E1_wordnet", "E2_embed", "E3_simplify", "E6_simplify_constrained"]}


def _english_rows(corpus: str) -> list[list[str]]:
    r = res()
    rows = []
    for v, systems in ENGLISH_ORDER.items():
        b = row(r, corpus, v, "E0_orig")
        for s in systems:
            x = row(r, corpus, v, s)
            if x is None:  # E5, E6c and E6r were run on the written sentences only
                continue
            ch = "" if s == "E0_orig" else f" ({signed(100 * (x.time_s / b.time_s - 1), 0)})"
            cells = [DICT[v], f"{CODE[s]} {NAME[s]}", ci(x.fs_rate, x.fs_rate_ci, k=100), f"{x.time_s:.1f}{ch}"]
            if corpus == "flores_eng":
                cells += [ci(x.comet, x.comet_ci, nd=3), f"{x.chrf:.1f}"]
            else:
                cells += [f"{x.cosine:.3f}", ci_or_value(x.entail, x.entail_ci, k=100)]
            cells.append(f"{x.latency:.2f}")
            rows.append(cells)
    return rows


def table7() -> str:
    heads = _heading_copies()
    return md_table("Table 7. English written sentences, fingerspelling, signing time, meaning and computing time of each method.",
                    ["Dictionary", "Method", "Content words fingerspelled, % (95% CI)", "Signing time, s (change, %)",
                     "COMET (95% CI)", "chrF", "Time per sentence, s"], _english_rows("flores_eng"),
                    "The 100 FLORES-200 sentences on which every method was run. COMET and chrF compare the output with "
                    "the English reference, which is the input itself. COMET also reads a source sentence, for which "
                    "the Greek version of each sentence was used. E3 does not use the dictionary, so its meaning scores "
                    "and its time are the same for every dictionary. For "
                    f"{heads[('flores_eng', 'E4_hints__is', 'only')]} of the 100 sentences, the output of E4 consists "
                    "only of the heading of its list of suggested words, Words that cannot be signed, because only the "
                    "first line of each answer is kept, and "
                    f"{heads[('flores_eng', 'E5_hints_constrained__is', 'any')]} outputs of E5 contain this heading. "
                    "Such outputs lower the meaning scores of E4 and make its fingerspelling and signing time look "
                    "lower than they are.")


def table8() -> str:
    return md_table("Table 8. English meeting segments, fingerspelling, signing time, meaning and computing time of each method.",
                    ["Dictionary", "Method", "Content words fingerspelled, % (95% CI)", "Signing time, s (change, %)",
                     "Cosine similarity with the original", "Mutual entailment with the original, % (95% CI)",
                     "Time per segment, s"], _english_rows("ami"),
                    "200 meeting segments. The meetings have no reference rewrites, so each output is compared with "
                    "the original segment. E3 does not use the dictionary, so its meaning scores and its time are the "
                    f"same for every dictionary. For {_heading_copies()[('ami', 'E4_hints__is', 'only')]} of the 200 "
                    "segments, the output of E4 consists only of the heading of its list of suggested words, which "
                    "lowers its meaning scores and makes its fingerspelling and signing time look lower than they "
                    "are.")


# ----------------------------------------------------------------------------------- Table 9
def table9() -> str:
    df = pd.read_csv(SCORES / "sentences.csv")
    ids = set(json.loads((ROOT / "data" / "manifest.json").read_text())["flores_llm_ids"])
    df = df[(df.corpus == "ami") | df["id"].astype(str).map(lambda x: x.isdigit() and int(x) in ids)]
    spec = [("flores_eng", "is", "E2_embed", "E1_wordnet", "comet"),
            ("flores_eng", "is", "E6_simplify_constrained", "E2_embed", "comet"),
            ("flores_eng", "is", "E4_hints", "E3_simplify", "comet"),
            ("flores_eng", "is", "E6_simplify_constrained", "E5_hints_constrained", "comet"),
            ("flores_eng", "is", "E6_simplify_constrained", "E6nofilter_simplify_constrained", "comet"),
            ("flores_eng", "is", "R5_E6_simplify_constrained", "E6_simplify_constrained", "comet")]
    for v in ["freq1000", "freq5000"]:
        spec += [("flores_eng", v, "E2_embed", "E1_wordnet", "comet"),
                 ("flores_eng", v, "E6_simplify_constrained", "E2_embed", "comet")]
    for v in ["is", "freq1000", "freq5000"]:
        spec += [("ami", v, "E2_embed", "E1_wordnet", "mutual_entailment"),
                 ("ami", v, "E6_simplify_constrained", "E2_embed", "mutual_entailment")]
    for c in ["flores_ell_Grek", "flores_fra_Latn"]:
        for v in ["is", "freq1000", "freq5000"]:
            spec += [(c, v, "T4_eurollm_constrained", "T6_nllb_simplify_constrained", "comet"),
                     (c, v, "T4_eurollm_constrained", "T6_nllb_simplify_constrained", "chrf")]
    measure = {"comet": "COMET", "mutual_entailment": "Mutual entailment, points", "chrf": "chrF per sentence"}
    rows = []
    for corpus, v, a, b, col in spec:
        sa = a if a == "E3_simplify" else f"{a}__{v}"
        sb = b if b == "E3_simplify" else f"{b}__{v}"
        d = paired_diff(df, corpus, v, sa, sb, col)
        k = 100 if col == "mutual_entailment" else 1
        nd = 3 if col == "comet" else 1
        rows.append([TEXT[corpus], DICT[v], f"{CODE[a]} minus {CODE[b]}", measure[col],
                     f"{signed(k * d['mean'], nd)} ({signed(k * d['lo'], nd)} to {signed(k * d['hi'], nd)})"])
    return md_table("Table 9. Differences in meaning between two methods on the same sentences.",
                    ["Text", "Dictionary", "Comparison", "Measure", "Mean difference (95% CI)"], rows,
                    "Mean of the difference between two methods, sentence by sentence, with a paired 95% bootstrap "
                    "confidence interval. A positive value means that the first method keeps more of the meaning, or, "
                    "for chrF, stays closer to the wording of the reference. An interval that excludes zero indicates "
                    "a difference beyond chance. Written, the 100 FLORES-200 sentences on which every method was run. "
                    "Meetings, 200 segments.")


# ----------------------------------------------------------------------------------- Tables 10 to 12
TRANSLATION = ["T0_nllb", "T1_eurollm", "T2_nllb_embed", "T6_nllb_simplify_constrained", "T4_eurollm_constrained"]


def table10() -> str:
    r = res()
    rows = []
    for v in ["is", "freq1000", "freq5000"]:
        for s in TRANSLATION:
            cells = [DICT[v], f"{CODE[s]} {NAME[s]}"]
            for c in ["flores_ell_Grek", "flores_fra_Latn"]:
                x = row(r, c, v, s)
                cells += [f"{100 * x.fs_rate:.1f}", f"{x.comet:.3f}", f"{x.chrf:.1f}"]
            rows.append(cells)
    return md_table("Table 10. Signable English from Greek and French, fingerspelling and meaning.",
                    ["Dictionary", "Method", "Greek, content words fingerspelled, %", "Greek, COMET", "Greek, chrF",
                     "French, content words fingerspelled, %", "French, COMET", "French, chrF"], rows,
                    "The 100 FLORES-200 sentences on which every method was run. COMET, which also reads the source "
                    "sentence, and chrF compare the output with the English reference. T0 and T1 do not use the "
                    "dictionary, so only their fingerspelling changes with it. The same sentences in English are "
                    "fingerspelled as shown for E0 in Table 7.")


def table11() -> str:
    r = res()
    rows = []
    for s in ["T0_nllb", "T1_eurollm", "T2_nllb_embed", "T6_nllb_simplify_constrained"]:
        cells = [f"{CODE[s]} {NAME[s]}"]
        for c in ["flores_spa_Latn", "flores_deu_Latn"]:
            x = row(r, c, "is", s)
            cells += [f"{100 * x.fs_rate:.1f}", f"{x.comet:.3f}", f"{x.chrf:.1f}"]
        rows.append(cells)
    return md_table("Table 11. Signable English from Spanish and German with the IS dictionary.",
                    ["Method", "Spanish, content words fingerspelled, %", "Spanish, COMET", "Spanish, chrF",
                     "German, content words fingerspelled, %", "German, COMET", "German, chrF"], rows,
                    "The 100 FLORES-200 sentences on which every method was run. For these two languages, T3 was run "
                    "with the IS dictionary only, and T4 was not run.")


def table12() -> str:
    r = res()
    lat = pd.read_csv(TABLES / "nllb_latency.csv").set_index("lang")
    langs = [("flores_ell_Grek", "ell_Grek"), ("flores_fra_Latn", "fra_Latn"), ("flores_spa_Latn", "spa_Latn"),
             ("flores_deu_Latn", "deu_Latn")]

    def cells(s, v):
        out = []
        for c, _ in langs:
            x = row(r, c, v, s)
            out.append(DASH if x is None else f"{x.latency:.2f}")
        return out

    rows = [["T0 NLLB translation, in batches of 16 sentences", DASH] + cells("T0_nllb", "is"),
            ["T0 NLLB translation, one sentence at a time", DASH] + [f"{lat.loc[code, 'mean']:.2f}" for _, code in langs],
            ["T1 EuroLLM translation", DASH] + cells("T1_eurollm", "is"),
            ["T2 NLLB, then embedding replacement", DICT["is"]] + cells("T2_nllb_embed", "is")]
    for s in ["T6_nllb_simplify_constrained", "T4_eurollm_constrained"]:
        for v in ["is", "freq1000", "freq5000"]:
            rows.append([f"{CODE[s]} {NAME[s]}", DICT[v]] + cells(s, v))
    return md_table("Table 12. Computing time per sentence of the translation methods.",
                    ["Method", "Dictionary", "Greek, s", "French, s", "Spanish, s", "German, s"], rows,
                    "Mean wall time per sentence on two CPU threads, for the 100 FLORES-200 sentences on which every "
                    "method was run, except NLLB one sentence at a time, which was timed separately and gave the same "
                    "output as batch translation for 98 to 99 of the 100 sentences. T2 and T3 include NLLB in batches. "
                    "The time of T4 does not include the unrestricted EuroLLM translation (T1) from which its names "
                    "and numbers are taken. In the Dictionary column, a dash means that no dictionary is used, and "
                    "elsewhere it marks a combination that was not run.")


# ----------------------------------------------------------------------------------- Table 13
def table13() -> str:
    pre = pd.read_csv(ERR / "prefix_substitution.csv")
    leak = pd.read_csv(ERR / "country_leaks.csv")
    drop = pd.read_csv(ERR / "dropped_stop_words.csv")
    spec = [("flores_eng", "is", s) for s in ["E1_wordnet__is", "E2_embed__is", "E3_simplify", "E4_hints__is",
                                              "E6_simplify_constrained__is", "E5_hints_constrained__is",
                                              "E6nofilter_simplify_constrained__is", "R5_E6_simplify_constrained__is"]]
    spec += [("flores_eng", v, f"{s}__{v}") for v in ["freq1000", "freq5000"]
             for s in ["E2_embed", "E6_simplify_constrained"]]
    spec += [("ami", "is", s) for s in ["E2_embed__is", "E3_simplify", "E6_simplify_constrained__is"]]
    spec += [(c, v, f"{s}__{v}") for c in ["flores_ell_Grek", "flores_fra_Latn"] for v in ["is", "freq1000", "freq5000"]
             for s in ["T2_nllb_embed", "T6_nllb_simplify_constrained", "T4_eurollm_constrained"]]
    spec += [(c, "is", f"{s}__is") for c in ["flores_spa_Latn", "flores_deu_Latn"]
             for s in ["T2_nllb_embed", "T6_nllb_simplify_constrained"]]
    rows = []
    for corpus, v, s in spec:
        p = pre[(pre.corpus == corpus) & (pre.vocab == v) & (pre.system == s)].iloc[0]
        lk = leak[(leak.corpus == corpus) & (leak.system == s)]
        dr = drop[(drop.corpus == corpus) & (drop.system == s)]
        rows.append([TEXT[corpus], DICT[v], CODE[base(s)], f"{100 * p.rate:.1f}", f"{100 * p.sentences_with_one:.0f}",
                     f"{100 * lk.iloc[0].share:.0f}" if len(lk) else DASH,
                     f"{int(dr.iloc[0].in_output)} / {int(dr.iloc[0].in_rewritten_text)}" if len(dr) else DASH])
    return md_table("Table 13. Failures of restricted rewriting, look-alike words, unwanted country names and dropped words.",
                    ["Text", "Dictionary", "Method", "Look-alike words, % of the words that would have been fingerspelled",
                     "Outputs with at least one look-alike word, %", "Outputs with an unwanted country name, %",
                     "Dropped stop-list words, output / text rewritten"], rows,
                    "A look-alike word is an output word that is not in the text being rewritten, begins with the same "
                    "three letters as one of the words that text would fingerspell and is not a form or derivative of "
                    "it (dinosaur became dinner). The methods without restriction show how often such pairs arise by "
                    "chance, for example from synonyms that share their first letters. The text being rewritten is the "
                    "input for English, NLLB's translation for T2 and T3, and EuroLLM's unrestricted translation for "
                    "T4. Unwanted country names were counted for the restricted methods, and dropped words for E3, E6 "
                    "and E6r and, for Greek and French, T3 and T4, both with the IS dictionary only. A dash marks a "
                    "measure that was not counted. Written, 100 FLORES-200 sentences. Meetings, 200 segments.")


# ----------------------------------------------------------------------------------- Table 14
def table14() -> str:
    from run import PER_CLIP
    from sgt.signing import plan
    from sgt.vocab import get_vocabulary
    vis = get_vocabulary("is", PER_CLIP)
    v5 = get_vocabulary("freq5000", PER_CLIP)
    recs = {}
    for s in ["E0_orig", "E2_embed__is", "E3_simplify", "E6_simplify_constrained__is",
              "R5_E6_simplify_constrained__is", "E6_simplify_constrained__freq5000"]:
        recs[s] = {json.loads(line)["id"]: json.loads(line)["output"]
                   for line in (OUT / "flores_eng" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()}
    rows = []
    for i in [379, 112]:
        for s, v, lab in [("E0_orig", vis, "IS"), ("E2_embed__is", vis, "IS"), ("E3_simplify", vis, "IS"),
                          ("E6_simplify_constrained__is", vis, "IS"), ("R5_E6_simplify_constrained__is", vis, "IS"),
                          ("E6_simplify_constrained__freq5000", v5, "Top 5,000 words")]:
            out = recs[s][i]
            p = plan(out, v)
            rows.append([CODE[base(s)], lab, out.replace("|", "/"), ", ".join(p.oov_words) if p.oov_words else "none"])
    return md_table("Table 14. Outputs of the English methods for two written sentences.",
                    ["Method", "Dictionary", "Output", "Content words fingerspelled"], rows,
                    "FLORES-200 devtest sentences 380 and 113 (line numbers of the devtest file, CC BY-SA 4.0). The "
                    "fingerspelled content words are those without a sign in the dictionary of the second column, "
                    "according to the signing model. Names and numbers are not listed.")


ALL = [table1, table2, table3, table4, table5, table6, table7, table8, table9, table10, table11, table12, table13,
       table14]


def main():
    parts = ["# Tables"] + [t() for t in ALL]
    text = "\n\n".join(parts) + "\n"
    out = SRC / "06_tables.md" if SRC.is_dir() else TABLES / "article_tables.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
