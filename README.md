# Vocabulary-constrained rewriting and translation for dictionary-based sign language avatars

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23396881.svg)](https://doi.org/10.5281/zenodo.23396881)

Code of the study "Say it so the avatar can sign it, rewriting and translating text into the vocabulary of a
dictionary-based sign language avatar" by Nikolaos D. Tantaroudas, Andrew J. McCracken, Ilias Karachalios and
Evangelos Papatheou.

A dictionary-based sign language avatar plays a recorded sign for each word in its dictionary and fingerspells the
other content words, letter by letter, which takes longer than signing. The study measures how much an International
Sign (IS) avatar with several hundred signs fingerspells in English meeting transcripts (AMI Meeting Corpus) and
written sentences (FLORES-200), and whether rewording the text, or translating text from Greek, French, Spanish and
German, into words that the avatar can sign avoids the fingerspelling without changing the meaning. Lists of the 500,
1,000, 2,000 and 5,000 most frequent English content words stand in for larger dictionaries. Every model runs on an
ordinary processor (two threads, 8 GB of memory), and no human participants were involved.

## Related outputs

| Output | Link |
|---|---|
| Data deposit (system outputs, scores, tables and failure analyses, CC BY 4.0 and CC BY-SA 4.0) | https://doi.org/10.5281/zenodo.23396571 |
| Preprint, rewriting English text | https://doi.org/10.20944/preprints202610.1279.v1 |
| Preprint, translating from Greek, French, Spanish and German | https://doi.org/10.20944/preprints202610.1475.v1 |
| Archived release of this software, version 1.0.0 | https://doi.org/10.5281/zenodo.23396881 |
| All versions of this software | https://doi.org/10.5281/zenodo.23396880 |

## Methods

| Code | Output file | Method |
|---|---|---|
| E0 | `E0_orig` | The English text signed as it is |
| E1 | `E1_wordnet__<dictionary>` | Each content word without a sign replaced by a WordNet synonym or more general word that has one |
| E2 | `E2_embed__<dictionary>` | Each content word without a sign replaced by the most similar word that has one, judged by word vectors |
| E3 | `E3_simplify` | Simplification by EuroLLM-1.7B-Instruct, unrestricted |
| E4 | `E4_hints__is` | Rewriting by EuroLLM-1.7B-Instruct with suggested replacement words, unrestricted |
| E5 | `E5_hints_constrained__is` | E4 restricted to words with a sign |
| E6 | `E6_simplify_constrained__<dictionary>` | E3 restricted to words with a sign |
| E6c | `E6nofilter_simplify_constrained__is` | E6 with every country name of the dictionary allowed |
| E6r | `R5_E6_simplify_constrained__is` | The best of five E6 rewrites, the one closest in meaning to the input (the four extra rewrites are in `S4_E6_simplify_constrained__is`) |
| T0 | `T0_nllb` | Translation into English by NLLB-200-distilled-600M |
| T1 | `T1_eurollm` | Translation into English by EuroLLM-1.7B-Instruct |
| T2 | `T2_nllb_embed__<dictionary>` | NLLB translation, then E2 |
| T3 | `T6_nllb_simplify_constrained__<dictionary>` | NLLB translation, then restricted simplification (E6) |
| T4 | `T4_eurollm_constrained__<dictionary>` | Direct translation by EuroLLM, restricted to words with a sign |

`<dictionary>` is `is` (the IS dictionary) or `freq500`, `freq1000`, `freq2000` and `freq5000`. The restriction is a
llama.cpp grammar that allows only the words of the dictionary in their grammatical forms. The code also contains two
restricted translation methods with suggested words (`T3_nllb_constrained`, `T5_eurollm_hints_constrained`) that were
tried on a few sentences and dropped, and they are not part of the results.

## Contents

| Path | What it is |
|---|---|
| `code/run.py` | Experiment runner, with the steps `prepare`, `nllb`, `swaps`, `llm` (a resumable job queue) and `sample` (the extra rewrites of E6r) |
| `code/sgt/vocab.py` | Dictionaries (the IS dictionary and the most frequent English content words), clip lengths and the test for country names |
| `code/sgt/signing.py` | Signing plan of a dictionary avatar, with its signs, fingerspelled words and letters and estimated signing time |
| `code/sgt/swaps.py` | Word replacement by WordNet and by word vectors, and the suggested words for E4 and E5 |
| `code/sgt/gbnf.py` | llama.cpp grammars that restrict generation to a list of allowed words |
| `code/sgt/llm.py` | EuroLLM-1.7B-Instruct on the processor (llama.cpp), prompts for translation and rewriting |
| `code/sgt/nllb.py` | NLLB-200-distilled-600M on the processor (CTranslate2, int8) |
| `code/sgt/corpora.py` | FLORES-200 devtest sentences and AMI meeting segments, fixed samples |
| `code/rerank.py` | E6r, the best of five restricted rewrites by sentence-embedding similarity to the input |
| `code/comet_score.py` | COMET (wmt22-comet-da), run in a separate environment |
| `code/score.py` | Signing, meaning and computing-time scores per sentence and per method |
| `code/analysis.py` | Result tables with bootstrap confidence intervals and paired differences |
| `code/lexicon_growth.py` | Fingerspelling when the IS dictionary is extended with frequent words, and the ranked list of the words it lacks |
| `code/errors.py` | Failure analysis, look-alike words, unwanted country names and dropped stop-list words |
| `code/nllb_latency.py` | NLLB time when it translates one sentence at a time |
| `code/retime.sh`, `code/retime_merge.py` | Repeated runs of two jobs whose timing was disturbed by another process (identical outputs, times replaced, first-run times kept as `seconds_first_run`) |
| `code/post_run.sh` | The steps after the language-model queue, in order |
| `code/pipeline_figure.py`, `code/figures.py` | Figure 1 (how a sentence reaches the avatar) and Figure 2 (fingerspelling removed against meaning kept) |
| `code/paper_tables.py` | Tables 1 to 14 of the article, generated from the result files |
| `data/manifest.json` | The sampled sentences and segments (seed 42), dictionary sizes and typical clip lengths |
| `data/per_clip_stats.csv` | Glosses and frame counts of the 747 clips of the IS dictionary, computed from its hand-tracking files |

## Installation

Two Python environments are needed, because COMET requires older versions of numpy and transformers.

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt     # Python 3.13, CPU only
#   torch as a CPU wheel, pip install torch --index-url https://download.pytorch.org/whl/cpu
#   spaCy models en_core_web_sm 3.8.0 and en_core_web_lg 3.8.0 (GitHub releases of explosion/spacy-models)
venv/bin/python -c "import nltk; nltk.download('wordnet')"
python3.12 -m venv venv_comet && venv_comet/bin/pip install -r requirements-comet.txt
```

Models, in `models/` (or set `EUROLLM_GGUF` and `NLLB_CT2`):

- `EuroLLM-1.7B-Instruct.Q8_0.gguf` (mradermacher/EuroLLM-1.7B-Instruct-GGUF on Hugging Face)
- NLLB-200-distilled-600M converted to CTranslate2,
  `ct2-transformers-converter --model facebook/nllb-200-distilled-600M --quantization int8 --output_dir models/nllb-200-distilled-600M-ct2-int8 --copy_files tokenizer.json sentencepiece.bpe.model tokenizer_config.json special_tokens_map.json`

Texts, in `data/raw/`:

- FLORES-200, https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz, extracted as `data/raw/flores200_dataset`
- AMI Meeting Corpus, the speaker segments of the 16 evaluation meetings from the manual annotations 1.6.2,
  as `data/raw/ami/*.json`

## Running the study

```bash
venv/bin/python code/run.py prepare
venv/bin/python code/run.py nllb
venv/bin/python code/run.py swaps
venv/bin/python code/run.py llm                       # about 4.5 hours on two threads, resumable
venv/bin/python code/run.py sample --corpora flores_eng --vocab is --n 4
venv/bin/python code/rerank.py
venv_comet/bin/python code/comet_score.py
venv/bin/python code/score.py && venv/bin/python code/analysis.py
venv/bin/python code/lexicon_growth.py && venv/bin/python code/errors.py && venv/bin/python code/nllb_latency.py
venv/bin/python code/pipeline_figure.py && venv/bin/python code/figures.py && venv/bin/python code/paper_tables.py
```

Do not run COMET and the language-model queue at the same time on a machine with 8 GB of memory. The tables are
written to `results/tables/article_tables.md` and the figures to `results/figures/`.

## Recomputing the tables and figures from the data deposit

Download the data deposit, unpack it and copy its folders `data` and `results` into the folder of this repository.
The scoring and analysis scripts (`code/score.py` and the scripts after it in the list above) then work from the saved
outputs and scores, without running the language and translation models again. They need the main Python environment
with the spaCy and WordNet data, and the texts in `data/raw/`.

## Data and licences

The code is released under the MIT licence (`LICENSE`). FLORES-200 is available under CC BY-SA 4.0 and the AMI Meeting
Corpus under CC BY 4.0, and the outputs derived from them in the data deposit carry the same licences.
EuroLLM-1.7B-Instruct is licensed under Apache 2.0 and wmt22-comet-da under Apache 2.0, while the weights of
NLLB-200-distilled-600M are licensed for non-commercial use only (CC BY-NC 4.0). Of the IS dictionary, only the glosses
and the frame counts of its clips are used (`data/per_clip_stats.csv`), computed from its openly available
hand-tracking files (https://doi.org/10.5281/zenodo.18656296). The hand-tracking files do not record the frame rate, so
frame counts are converted to seconds at 30 frames per second, the frame rate of five source clips that were checked.

## Funding

This work was supported by FSTP funding from the European Union's Horizon Europe research and innovation programme
under grant agreement No. 101070631 (UTTER, Unified Transcription and Translation for Extended Reality) and by UK
Research and Innovation (UKRI) under the UK government's Horizon Europe funding guarantee (grant No. 10039436). Views
and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or the
granting authority. Neither the European Union nor the granting authority can be held responsible for them.

## Citation

Please cite the article and this software as given in `CITATION.cff`.
