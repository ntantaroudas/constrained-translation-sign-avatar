"""NLLB-200-distilled-600M with CTranslate2 (int8, CPU), beam size 4."""
from __future__ import annotations

import time

import ctranslate2
from transformers import AutoTokenizer


class NLLB:
    def __init__(self, model_dir: str, threads: int = 2):
        self.tr = ctranslate2.Translator(model_dir, device="cpu", compute_type="int8",
                                         intra_threads=threads)
        self.model_dir = model_dir
        self._tok: dict[str, AutoTokenizer] = {}

    def tok(self, src_lang: str):
        if src_lang not in self._tok:
            self._tok[src_lang] = AutoTokenizer.from_pretrained(self.model_dir, src_lang=src_lang)
        return self._tok[src_lang]

    def translate(self, sents: list[str], src_lang: str, tgt_lang: str = "eng_Latn",
                  beam: int = 4, batch: int = 16) -> list[tuple[str, float]]:
        tk = self.tok(src_lang)
        out = []
        for i in range(0, len(sents), batch):
            chunk = sents[i:i + batch]
            toks = [tk.convert_ids_to_tokens(tk.encode(s)) for s in chunk]
            t0 = time.perf_counter()
            res = self.tr.translate_batch(toks, target_prefix=[[tgt_lang]] * len(chunk),
                                          beam_size=beam, max_decoding_length=256)
            dt = (time.perf_counter() - t0) / len(chunk)
            for r in res:
                hyp = r.hypotheses[0][1:]  # drop the target language token
                out.append((tk.decode(tk.convert_tokens_to_ids(hyp), skip_special_tokens=True), dt))
        return out
