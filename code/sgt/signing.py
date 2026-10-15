"""Signing plan of a dictionary-based avatar for an English sentence.

Each token becomes one of:
  sign    - a content word or multiword gloss in the vocabulary, played as one clip
  sign_fn - a stop word, pronoun or interjection that happens to be in the vocabulary
            (for example I or you in the IS lexicon): signed, but not counted as a
            content word, so that the share of content words fingerspelled is
            comparable between lexicons with and without such entries
  oov     - a content word without an entry, fingerspelled (avoidable)
  name    - a name without an entry, fingerspelled (also needed in human signing)
  number  - a number, signed with numeral clips
  drop    - a function word (determiner, adposition, conjunction, particle,
            auxiliary), a filler, an interjection or pronoun outside the
            vocabulary, or a stop word; not signed
  punct   - punctuation, ignored

The plan gives the counts used in the paper and an estimate of signing time:
clip lengths are the median active-segment lengths of the IS clips
(word signs, multiword signs, letters and numerals); for the IS vocabulary each
sign uses its own clip length.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from functools import lru_cache

import spacy

from .vocab import STOP, Vocabulary, lemma

FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "AUX"}  # not signed
NAME_ENTS = {"PERSON", "GPE", "LOC", "ORG", "NORP", "FAC", "EVENT", "WORK_OF_ART",
             "PRODUCT", "LANGUAGE", "LAW"}
FILLERS = {"uh", "um", "umm", "mm", "mmm", "hmm", "mm-hmm", "uh-huh", "huh", "er", "erm",
           "ah", "eh", "oh", "mhm", "uhm", "hm"}
NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
                "billion", "first", "second", "third"}


@lru_cache(maxsize=1)
def nlp():
    return spacy.load("en_core_web_sm", disable=["parser"])


@dataclass
class Plan:
    n_sign: int = 0       # signed content words and multiword glosses
    n_sign_fn: int = 0    # signed stop words, pronouns and interjections
    n_oov: int = 0
    n_name: int = 0
    n_number: int = 0
    n_drop: int = 0
    letters_oov: int = 0
    letters_name: int = 0
    frames: float = 0.0
    oov_words: tuple = ()
    signed_lemmas: tuple = ()

    @property
    def content(self) -> int:
        return self.n_sign + self.n_oov

    def as_dict(self) -> dict:
        d = asdict(self)
        d["oov_words"] = list(self.oov_words)
        d["signed_lemmas"] = list(self.signed_lemmas)
        d["content"] = self.content
        return d


def _letters(s: str) -> int:
    return sum(ch.isalpha() for ch in s)


def tokens_with_roles(text: str, vocab: Vocabulary):
    """Yield (token, role, matched_lemma) for every token of `text`."""
    doc = nlp()(text)
    toks = [t for t in doc if not t.is_space]
    lowers = [t.lower_ for t in toks]
    lems = [t.lemma_.lower() for t in toks]
    in_mw = [None] * len(toks)
    for mw in vocab.multiword:                       # longest first
        n = len(mw)
        for i in range(len(toks) - n + 1):
            if any(in_mw[i:i + n]):
                continue
            if tuple(lowers[i:i + n]) == mw or tuple(lems[i:i + n]) == mw:
                for j in range(i, i + n):
                    in_mw[j] = (i, mw)
    for i, t in enumerate(toks):
        if in_mw[i] is not None:
            start, mw = in_mw[i]
            yield t, ("sign_mw" if start == i else "mw_cont"), " ".join(mw)
            continue
        txt = t.text
        low = t.lower_
        if not any(ch.isalnum() for ch in txt):
            yield t, "punct", None
            continue
        if low in FILLERS:
            yield t, "drop", None
            continue
        if t.like_num or re.fullmatch(r"[\d.,]+", txt) or low in NUMBER_WORDS:
            yield t, "number", None
            continue
        if t.pos_ in FUNCTION_POS or low in {"'s", "’s", "n't", "n’t"}:
            yield t, "drop", None
            continue
        m = vocab.is_signable(low, t.lemma_.lower(), lemma(low))
        function_like = low in STOP or t.lemma_.lower() in STOP or t.pos_ in {"INTJ", "PRON"}
        if m:
            yield t, ("sign_fn" if function_like else "sign"), m
            continue
        if function_like:
            yield t, "drop", None
            continue
        if t.pos_ == "PROPN" or t.ent_type_ in NAME_ENTS:
            yield t, "name", None
            continue
        yield t, "oov", None


def plan(text: str, vocab: Vocabulary) -> Plan:
    p = Plan()
    oov, signed = [], []
    d = vocab.durations
    for t, role, m in tokens_with_roles(text, vocab):
        if role in ("sign", "sign_fn"):
            if role == "sign":
                p.n_sign += 1
            else:
                p.n_sign_fn += 1
            p.frames += vocab.sign_frames(m)
            signed.append(m)
        elif role == "sign_mw":
            p.n_sign += 1
            p.frames += d.multiword
            signed.append(m)
        elif role == "oov":
            p.n_oov += 1
            n = _letters(t.text)
            p.letters_oov += n
            p.frames += n * d.letter
            oov.append(t.lower_)
        elif role == "name":
            p.n_name += 1
            n = _letters(t.text)
            p.letters_name += n
            p.frames += n * d.letter
        elif role == "number":
            p.n_number += 1
            digits = sum(ch.isdigit() for ch in t.text) or 1
            p.frames += digits * d.number
        elif role == "drop":
            p.n_drop += 1
    p.oov_words = tuple(oov)
    p.signed_lemmas = tuple(signed)
    return p
