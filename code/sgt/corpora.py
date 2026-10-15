"""Evaluation sentences.

* FLORES-200 devtest (CC BY-SA 4.0): English, Greek, French, Spanish, German.
* AMI meeting corpus, evaluation meetings (CC BY 4.0): speaker segments with
  fillers removed, 6-40 words and at least three content words.
Fixed random samples (seed 42) of 200 sentences and 200 segments are drawn; the LLM
systems use 100 of the 200 FLORES sentences (seed 7, see run.py prepare).
"""
from __future__ import annotations

import glob
import json
import random
import re
from pathlib import Path

from .signing import FILLERS
from .vocab import STOP

FLORES_LANGS = ["eng_Latn", "ell_Grek", "fra_Latn", "spa_Latn", "deu_Latn"]
SEED = 42


def load_flores(root: Path, split: str = "devtest") -> dict[str, list[str]]:
    return {lang: (root / split / f"{lang}.{split}").read_text(encoding="utf-8").splitlines()
            for lang in FLORES_LANGS}


def _clean_ami(text: str) -> str:
    words = [w for w in text.split() if w.lower().strip(".,?!") not in FILLERS]
    out = " ".join(words)
    out = re.sub(r"\b(\w+)( \1\b)+", r"\1", out, flags=re.IGNORECASE)  # "I I just" -> "I just"
    return out.strip()


def load_ami(ref_dir: Path) -> list[dict]:
    items = []
    for f in sorted(glob.glob(str(ref_dir / "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for k, s in enumerate(d["segments"]):
            text = _clean_ami(s["text"])
            words = text.split()
            content = [w for w in words if w.isalpha() and w.lower() not in STOP and len(w) > 1]
            if 6 <= len(words) <= 40 and len(content) >= 3:
                items.append({"id": f"{d['meeting']}:{k}", "text": text[:1].upper() + text[1:]})
    return items


def sample_ids(n_total: int, n: int, seed: int = SEED) -> list[int]:
    return sorted(random.Random(seed).sample(range(n_total), n))
