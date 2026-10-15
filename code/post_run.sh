#!/usr/bin/env bash
# After the LLM queue: COMET and AMI meaning scores, best-of-5 sampling, reranking,
# final scoring, tables and figure. Steps that load large models are not run in
# parallel with the LLM (8 GB machine).
set -u
cd "$(dirname "$0")/.."
QUEUE_PID=${1:-}
if [ -n "$QUEUE_PID" ]; then
  while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 30; done
fi
echo "[$(date +%T)] queue finished"
venv_comet/bin/python code/comet_score.py > results/comet_run2.log 2>&1 &
C=$!
SCORE_ONLY="ami/" venv/bin/python code/score.py > results/score_ami2.log 2>&1
wait $C
echo "[$(date +%T)] COMET and AMI meaning done"
venv/bin/python -u code/run.py sample --vocab is --corpora flores_eng --n 4 > results/sample_run.log 2>&1
echo "[$(date +%T)] sampling done"
venv/bin/python code/rerank.py > results/rerank.log 2>&1
venv_comet/bin/python code/comet_score.py > results/comet_run3.log 2>&1
venv/bin/python code/score.py > results/score_all.log 2>&1
venv/bin/python code/analysis.py > results/analysis.log 2>&1
venv/bin/python code/figures.py > results/figures.log 2>&1
# added after the first run: failure analysis, NLLB one-sentence latency, article tables
venv/bin/python code/errors.py > results/errors.log 2>&1
venv/bin/python code/nllb_latency.py > results/nllb_latency.log 2>&1
venv/bin/python code/paper_tables.py > results/paper_tables.log 2>&1
echo "[$(date +%T)] all done"
