#!/usr/bin/env bash
# Re-measure the generation time of two runs whose timing was disturbed by another job
# running at the same time (E5 on English FLORES, EuroLLM translation of Greek).
# Outputs are regenerated with the same settings and must be identical; only the times
# are replaced (code/retime_merge.py keeps the first-run times as seconds_first_run).
set -eu
cd "$(dirname "$0")/.."
venv/bin/python -u code/run.py llm --only "E5_hints_constrained__is@flores_eng" --model-tag retime
venv/bin/python -u code/run.py llm --only "T1_eurollm@flores_ell_Grek" --model-tag retime
venv/bin/python code/retime_merge.py
