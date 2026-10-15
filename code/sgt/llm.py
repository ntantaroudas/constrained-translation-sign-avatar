"""EuroLLM (GGUF, llama.cpp, CPU) for translation and vocabulary-restricted rewriting.

Prompts follow the official EuroLLM chat template; translation uses the prompt
from the EuroLLM model card. Decoding is greedy (temperature 0).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from llama_cpp import Llama, LlamaGrammar

from .gbnf import SentenceGrammar

LANG_NAMES = {"ell_Grek": "Greek", "fra_Latn": "French", "spa_Latn": "Spanish",
              "deu_Latn": "German", "eng_Latn": "English"}

PROMPT_TRANSLATE = "Translate the following {lang} source text to English:\n{lang}: {src} \nEnglish: "

PROMPT_SIMPLIFY = ("Rewrite the following sentence in simple, everyday English, using only very "
                   "common words. Keep all names and numbers. Keep the meaning. Reply with the "
                   "rewritten sentence only.\nSentence: {text}")

PROMPT_HINTS = ("Rewrite the sentence below for a sign-language avatar that can only sign a "
                "limited vocabulary. Some of its words cannot be signed; suggested replacements "
                "are given for each. Replace those words, or rephrase that part with very common "
                "words. Do not add new information. Keep all names and numbers. Reply with the "
                "rewritten sentence only.\nWords that cannot be signed:\n{hints}\n"
                "Sentence: {text}")

PROMPT_TRANSLATE_HINTS = ("Translate the following {lang} source text to English for a sign-language "
                          "avatar that can only sign a limited vocabulary. Use very common words. "
                          "These English words cannot be signed, so use the suggested replacements "
                          "or rephrase:\n{hints}\nDo not add new information. Keep all names and "
                          "numbers.\n{lang}: {src} \nEnglish: ")


REPEAT_PENALTY = 1.1  # llama.cpp's customary value; same for every system


def chatml(user: str, system: str = "") -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def hint_lines(hints: list[tuple[str, list[str]]]) -> str:
    return "\n".join(f"- {w}: {', '.join(c) if c else '(rephrase)'}" for w, c in hints)


def clean(text: str) -> str:
    t = text.strip().split("\n")[0].strip()
    for pre in ("Rewritten sentence:", "Sentence:", "English:", "Answer:"):
        if t.lower().startswith(pre.lower()):
            t = t[len(pre):].strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    return t


@dataclass
class Gen:
    text: str
    raw: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    finish_reason: str


class EuroLLM:
    def __init__(self, model_path: str, n_threads: int = 2, n_ctx: int = 2048):
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_batch=256, seed=0, verbose=False)
        self._grammars: dict[str, SentenceGrammar] = {}

    def generate(self, prompt: str, grammar: LlamaGrammar | None = None,
                 max_tokens: int = 128, temperature: float = 0.0, seed: int | None = None) -> Gen:
        t0 = time.perf_counter()
        kw = dict(temperature=0.0, top_k=1) if temperature == 0.0 else \
            dict(temperature=temperature, top_k=40, top_p=0.95, seed=seed)
        out = self.llm.create_completion(prompt=prompt, max_tokens=max_tokens,
                                         repeat_penalty=REPEAT_PENALTY,
                                         stop=["<|im_end|>"], grammar=grammar, **kw)
        dt = time.perf_counter() - t0
        raw = out["choices"][0]["text"]
        return Gen(clean(raw), raw, out["usage"]["prompt_tokens"],
                   out["usage"]["completion_tokens"], dt, out["choices"][0]["finish_reason"])

    def grammar(self, key: str, words, extra=()) -> LlamaGrammar:
        if key not in self._grammars:
            self._grammars[key] = SentenceGrammar(words)
        return LlamaGrammar.from_string(self._grammars[key].text(extra), verbose=False)

    @staticmethod
    def max_tokens_for(text: str) -> int:
        return max(48, int(len(text.split()) * 3) + 16)

    def translate(self, src: str, lang: str, grammar=None) -> Gen:
        p = chatml(PROMPT_TRANSLATE.format(lang=LANG_NAMES[lang], src=src))
        return self.generate(p, grammar, self.max_tokens_for(src))

    def translate_with_hints(self, src: str, lang: str, hints, grammar=None) -> Gen:
        p = chatml(PROMPT_TRANSLATE_HINTS.format(lang=LANG_NAMES[lang], src=src,
                                                 hints=hint_lines(hints)))
        return self.generate(p, grammar, self.max_tokens_for(src))

    def simplify(self, text: str, grammar=None, temperature: float = 0.0, seed=None) -> Gen:
        return self.generate(chatml(PROMPT_SIMPLIFY.format(text=text)), grammar,
                             self.max_tokens_for(text), temperature, seed)

    def rewrite_with_hints(self, text: str, hints, grammar=None) -> Gen:
        p = chatml(PROMPT_HINTS.format(text=text, hints=hint_lines(hints)))
        return self.generate(p, grammar, self.max_tokens_for(text))
