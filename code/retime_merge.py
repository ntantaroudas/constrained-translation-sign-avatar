#!/usr/bin/env python3
"""Replace the generation times of a run by those of its re-run (<system>@retime.jsonl),
after checking that the outputs are identical; the first-run times are kept as
seconds_first_run, and the re-run file is removed."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "outputs"
PAIRS = [("flores_eng", "E5_hints_constrained__is"), ("flores_ell_Grek", "T1_eurollm")]


def main():
    for corpus, system in PAIRS:
        f, g = OUT / corpus / f"{system}.jsonl", OUT / corpus / f"{system}@retime.jsonl"
        if not g.exists():
            print("missing", g); continue
        new = {json.loads(l)["id"]: json.loads(l) for l in g.read_text(encoding="utf-8").splitlines() if l.strip()}
        recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        same = diff = 0
        for r in recs:
            n = new.get(r["id"])
            if n is None:
                continue
            if n["output"] != r["output"]:
                diff += 1
                print("output differs:", corpus, system, r["id"])
                continue
            same += 1
            if "seconds_first_run" not in r:
                r["seconds_first_run"] = r["seconds"]
            r["seconds"] = n["seconds"]
        if diff:
            sys.exit(f"{corpus}/{system}: {diff} outputs differ; times not replaced")
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        g.unlink()
        print(f"{corpus}/{system}: {same} times replaced")


if __name__ == "__main__":
    main()
