"""Word-level substitution of words the avatar cannot sign.

* WordNet: synonyms, then hypernyms (two levels), then similar adjectives,
  in WordNet's sense order; the first candidate in the vocabulary is used.
* Embedding: nearest vocabulary lemma by cosine similarity of spaCy
  en_core_web_lg word vectors, used if the similarity reaches a threshold.
Candidates must have a WordNet sense with the same part of speech as the word
they replace, and are inflected like it (lemminflect).

`candidates()` merges both sources into a short list for the LLM prompts.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import spacy
from nltk.corpus import wordnet as wn

from lemminflect import getInflection

from .signing import tokens_with_roles
from .vocab import STOP, Vocabulary, is_country

WN_POS = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV, "PROPN": wn.NOUN}


@lru_cache(maxsize=1)
def nlp_md():
    return spacy.load("en_core_web_lg", disable=["parser", "ner", "tagger", "lemmatizer",
                                                  "attribute_ruler", "senter"])


def wordnet_candidates(word: str, pos: str | None) -> list[str]:
    wpos = WN_POS.get(pos or "")
    synsets = wn.synsets(word, pos=wpos) if wpos else wn.synsets(word)
    if not synsets and wpos:
        synsets = wn.synsets(word)
    out: list[str] = []

    def add(names):
        for n in names:
            n = n.lower().replace("_", " ")
            if n != word and n not in out:
                out.append(n)
    for s in synsets:
        add(s.lemma_names())
    for s in synsets:
        for h in s.hypernyms() + s.instance_hypernyms():
            add(h.lemma_names())
    for s in synsets:
        for h in s.hypernyms():
            for h2 in h.hypernyms():
                add(h2.lemma_names())
    for s in synsets:
        for sim in s.similar_tos():
            add(sim.lemma_names())
    return out


class EmbeddingIndex:
    """Cosine nearest neighbours among the vocabulary lemmas that have vectors."""

    def __init__(self, vocab: Vocabulary):
        nv = nlp_md().vocab
        words, vecs = [], []
        for w in sorted(vocab.lemmas):
            if w in STOP or not w.isalpha() or is_country(w):
                continue
            lex = nv[w]
            if lex.has_vector:
                words.append(w)
                vecs.append(lex.vector)
        m = np.asarray(vecs, dtype=np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
        self.words, self.matrix = words, m

    def neighbours(self, word: str, k: int = 5, pos: str | None = None) -> list[tuple[str, float]]:
        lex = nlp_md().vocab[word]
        if not lex.has_vector or not self.words:
            return []
        v = lex.vector / (np.linalg.norm(lex.vector) + 1e-8)
        sims = self.matrix @ v
        out = []
        for i in np.argsort(-sims):
            w = self.words[i]
            if w == word or not same_pos(w, pos):
                continue
            out.append((w, float(sims[i])))
            if len(out) >= k:
                break
        return out


def same_pos(word: str, pos: str | None) -> bool:
    """True if `word` has a WordNet sense with the part of speech `pos` (or pos unknown)."""
    wpos = WN_POS.get(pos or "")
    if wpos is None:
        return True
    if wpos == wn.ADJ:
        return bool(wn.synsets(word, pos=wn.ADJ) or wn.synsets(word, pos=wn.ADJ_SAT))
    return bool(wn.synsets(word, pos=wpos))


def inflect(cand: str, tag: str) -> str:
    if " " in cand or not tag:
        return cand
    forms = getInflection(cand, tag=tag)
    return forms[0] if forms else cand


def _in_vocab(cand: str, vocab: Vocabulary) -> bool:
    """A usable replacement: in the vocabulary and not a country name."""
    if is_country(cand):
        return False
    if " " in cand:
        return tuple(cand.split()) in set(vocab.multiword)
    return vocab.is_signable(cand) is not None


def candidates(word: str, lemma_: str, pos: str | None, vocab: Vocabulary,
               index: EmbeddingIndex, k: int = 4, min_sim: float = 0.45) -> list[str]:
    out: list[str] = []
    for c in wordnet_candidates(lemma_, pos):
        if len(out) >= k // 2:
            break
        if _in_vocab(c, vocab) and c not in out:
            out.append(c)
    for c, sim in index.neighbours(lemma_, k=k, pos=pos):
        if sim < min_sim:
            break
        if c not in out:
            out.append(c)
        if len(out) >= k:
            break
    return out


def _match_case(src: str, cand: str) -> str:
    return cand[:1].upper() + cand[1:] if src[:1].isupper() else cand


def substitute(text: str, vocab: Vocabulary, method: str, index: EmbeddingIndex | None = None,
               tau: float = 0.5) -> tuple[str, list[dict]]:
    """Replace each fingerspelled content word with a vocabulary word.
    method: 'wordnet' or 'embedding'. Returns (new_text, replacements)."""
    pieces, reps = [], []
    for t, role, _ in tokens_with_roles(text, vocab):
        new = t.text
        if role == "oov":
            cand = None
            if method == "wordnet":
                for c in wordnet_candidates(t.lemma_.lower(), t.pos_):
                    if _in_vocab(c, vocab):
                        cand = c
                        break
            elif method == "embedding":
                nb = index.neighbours(t.lemma_.lower(), k=1, pos=t.pos_) if index else []
                if not nb:
                    nb = index.neighbours(t.lower_, k=1, pos=t.pos_) if index else []
                if nb and nb[0][1] >= tau:
                    cand = nb[0][0]
            if cand:
                new = _match_case(t.text, inflect(cand, t.tag_))
                reps.append({"word": t.text, "replacement": cand})
        pieces.append(new + t.whitespace_)
    return "".join(pieces).strip(), reps
