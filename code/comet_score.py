#!/usr/bin/env python3
"""COMET (Unbabel/wmt22-comet-da) for every FLORES output file, on CPU.

Runs in its own environment (venv_comet: Python 3.12, unbabel-comet 2.2.7,
transformers 4.x, numpy 1.26), because unbabel-comet does not support
transformers 5 / numpy 2. Scores are cached per file and only new sentences
are scored.

source = the FLORES source sentence (Greek for the English-input corpus),
mt     = the system output, ref = the FLORES English reference.

    venv_comet/bin/python code/comet_score.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "outputs"
CACHE = ROOT / "results" / "comet"
FLORES = ROOT / "data" / "raw" / "flores200_dataset" / "devtest"


def lines(lang):
    return (FLORES / f"{lang}.devtest").read_text(encoding="utf-8").splitlines()


def main():
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da",
                                                saving_directory=str(ROOT / "models" / "comet")))
    ref = lines("eng_Latn")
    for corpus_dir in sorted(OUT.glob("flores_*")):
        corpus = corpus_dir.name
        lang = "ell_Grek" if corpus == "flores_eng" else corpus.replace("flores_", "")
        src = lines(lang)
        for f in sorted(corpus_dir.glob("*.jsonl")):
            if "@test" in f.name or re.match(r"S\d+_", f.stem):
                continue  # test runs; best-of-N candidate files (scored through their R file)
            cache_f = CACHE / corpus / (f.stem + ".json")
            cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
            recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            todo = [r for r in recs if str(r["id"]) not in cache]
            if not todo:
                continue
            data = [{"src": src[r["id"]], "mt": r["output"], "ref": ref[r["id"]]} for r in todo]
            out = model.predict(data, batch_size=16, gpus=0, progress_bar=False)
            for r, s in zip(todo, out.scores):
                cache[str(r["id"])] = float(s)
            cache_f.parent.mkdir(parents=True, exist_ok=True)
            cache_f.write_text(json.dumps(cache))
            print(corpus, f.stem, len(cache), flush=True)
    # identity reference point: the English reference scored against itself
    cache_f = CACHE / "flores_eng" / "REF_identity.json"
    if not cache_f.exists():
        ids = json.loads((ROOT / "data" / "manifest.json").read_text())["flores_ids"]
        src = lines("ell_Grek")
        out = model.predict([{"src": src[i], "mt": ref[i], "ref": ref[i]} for i in ids],
                            batch_size=16, gpus=0, progress_bar=False)
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps({str(i): float(s) for i, s in zip(ids, out.scores)}))


if __name__ == "__main__":
    sys.exit(main())
