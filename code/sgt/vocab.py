"""Vocabularies a dictionary-based signing avatar can sign, and clip durations.

Two kinds of vocabulary are used:

* ``is``      : the International Sign (IS) lexicon of ORE 6:72 (glosses with
                hand landmarks, read from per_clip_stats.csv). Only the gloss
                words and clip lengths are used here, never the clips.
* ``freqK``   : the K most frequent English content lemmas (K = 500 ... 5000),
                built exactly as in lexicon_coverage.py of ORE 6:72, which
                stands in for a lexicon of that size.

Matching is lemma-based: a running word is signable if its lower-cased form,
its spaCy lemma or its simplemma lemma is a vocabulary lemma.
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import simplemma
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordfreq import top_n_list, word_frequency

STOP = frozenset(ENGLISH_STOP_WORDS)
CANDIDATES = 60000  # wordfreq entries scanned (as in lexicon_coverage.py)


def lemma(word: str) -> str:
    return simplemma.lemmatize(word.lower(), lang="en").lower()


@lru_cache(maxsize=1)
def ranked_content_lemmas() -> list[tuple[str, float]]:
    """English content lemmas ranked by summed wordfreq frequency (stop words removed)."""
    freq: dict[str, float] = {}
    for w in top_n_list("en", CANDIDATES):
        if not (w.isascii() and w.isalpha() and len(w) >= 2) or w in STOP:
            continue
        lm = lemma(w)
        if lm in STOP or not lm.isalpha():
            continue
        freq[lm] = freq.get(lm, 0.0) + word_frequency(w, "en")
    return sorted(freq.items(), key=lambda kv: kv[1], reverse=True)


@lru_cache(maxsize=1)
def english_forms() -> dict[str, set[str]]:
    """lemma -> lower-case surface forms seen in the wordfreq list."""
    forms: dict[str, set[str]] = {}
    for w in top_n_list("en", CANDIDATES):
        if w.isascii() and w.isalpha():
            forms.setdefault(lemma(w), set()).add(w)
    return forms


@lru_cache(maxsize=1)
def _country_roots():
    return {wn.synset("country.n.02"), wn.synset("state.n.04")}


@lru_cache(maxsize=None)
def is_country(term: str) -> bool:
    """True if WordNet has a sense of `term` that is an instance of a country or state
    (e.g. Paraguay, Turkey, New Zealand)."""
    roots = _country_roots()
    for s in wn.synsets(term.replace(" ", "_"), pos=wn.NOUN):
        for ih in s.instance_hypernyms():
            if ih in roots or roots & set(ih.closure(lambda x: x.hypernyms())):
                return True
    return False


@dataclass
class Durations:
    """Median active-segment lengths (frames) of the IS clips, by clip type."""
    sign: float
    multiword: float
    letter: float
    number: float
    per_gloss: dict[str, float] = field(default_factory=dict)  # IS gloss or lemma -> frames


@dataclass
class Vocabulary:
    name: str
    lemmas: set[str]                              # signable single-word lemmas
    multiword: list[tuple[str, ...]]              # signable multiword glosses (lower-case tokens)
    durations: Durations
    use_gloss_durations: bool = False

    def is_signable(self, *candidates: str) -> str | None:
        """Return the matching vocabulary lemma, or None."""
        for c in candidates:
            if c and c in self.lemmas:
                return c
        return None

    def sign_frames(self, lm: str) -> float:
        if self.use_gloss_durations:
            return self.durations.per_gloss.get(lm, self.durations.sign)
        return self.durations.sign

    @property
    def countries(self) -> set[str]:
        """Country-name entries (single words and multiword glosses)."""
        return ({lm for lm in self.lemmas if is_country(lm)}
                | {" ".join(mw) for mw in self.multiword if is_country(" ".join(mw))})

    @property
    def allowed_forms(self) -> set[str]:
        """Words the restricted decoder may produce: surface forms of the single-word
        vocabulary lemmas, and each multiword gloss as a whole phrase (its words are not
        allowed on their own). Country names are left out: they are allowed only when
        the sentence itself mentions them (see run.extras_from), so that they cannot act
        as completions of unrelated words (Par... -> Paraguay)."""
        forms = english_forms()
        countries = self.countries
        out: set[str] = set()
        for lm in self.lemmas:
            if lm.isalpha() and lm not in countries:
                out.add(lm)
                out |= forms.get(lm, set())
        for mw in self.multiword:
            phrase = " ".join(mw)
            if all(t.isalnum() for t in mw) and phrase not in countries:
                out.add(phrase)
        return out


def load_durations(per_clip_csv: Path) -> tuple[Durations, dict]:
    rows = [r for r in csv.DictReader(open(per_clip_csv, encoding="utf-8"))
            if int(r["any_hand_frames"]) > 0]
    by_cat: dict[str, list[int]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(int(r["active_frames"]))
    per_gloss: dict[str, list[int]] = {}
    for r in rows:
        if r["category"] == "single_word":
            g = r["gloss"].strip().lower()
            # keyed by the gloss and by its lemma, since a token can match either
            for key in {g, lemma(g)}:
                per_gloss.setdefault(key, []).append(int(r["active_frames"]))
    d = Durations(
        sign=statistics.median(by_cat["single_word"]),
        multiword=statistics.median(by_cat["multiword"]),
        letter=statistics.median(by_cat["single_letter"]),
        number=statistics.median(by_cat["number"]),
        per_gloss={k: statistics.median(v) for k, v in per_gloss.items()},
    )
    return d, {"n_clips_with_hands": len(rows), **{f"n_{k}": len(v) for k, v in by_cat.items()}}


def is_vocabulary(per_clip_csv: Path) -> Vocabulary:
    d, _ = load_durations(per_clip_csv)
    lemmas, multi = set(), set()
    for r in csv.DictReader(open(per_clip_csv, encoding="utf-8")):
        if int(r["any_hand_frames"]) == 0:
            continue
        g = r["gloss"].strip().lower()
        if r["category"] == "single_word":
            lemmas.add(lemma(g))
            lemmas.add(g)
        elif r["category"] == "multiword":
            toks = tuple(g.replace("-", " ").split())
            if toks:
                multi.add(toks)
    return Vocabulary("is", lemmas, sorted(multi, key=len, reverse=True), d, use_gloss_durations=True)


def frequency_vocabulary(k: int, durations: Durations) -> Vocabulary:
    ranked = ranked_content_lemmas()
    return Vocabulary(f"freq{k}", {w for w, _ in ranked[:k]}, [], durations)


def get_vocabulary(name: str, per_clip_csv: Path) -> Vocabulary:
    if name == "is":
        return is_vocabulary(per_clip_csv)
    if name.startswith("freq"):
        d, _ = load_durations(per_clip_csv)
        return frequency_vocabulary(int(name[4:]), d)
    raise ValueError(name)
