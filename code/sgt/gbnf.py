"""llama.cpp GBNF grammars that restrict generation to an allowed word list.

Words are stored in a radix tree, so the grammar has few alternatives at each
character and llama.cpp's grammar sampler stays fast for >10,000 words.
"""
from __future__ import annotations


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _radix(words):
    root: dict = {}
    for w in words:
        node = root
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = True

    def compress(node):
        out = {}
        for ch, child in node.items():
            if ch == "$":
                continue
            label, cur = ch, child
            while True:
                keys = [k for k in cur if k != "$"]
                if len(keys) == 1 and "$" not in cur:
                    label += keys[0]
                    cur = cur[keys[0]]
                else:
                    break
            out[label] = ("$" in cur, compress(cur))
        return out
    return compress(root)


def word_rules(words, name: str) -> list[str]:
    """GBNF rules defining <name> as any word in `words`."""
    tree = _radix(sorted(set(words)))
    rules: list[str] = []
    counter = [0]

    def emit(subtree, rname):
        alts = []
        for label, (terminal, child) in sorted(subtree.items()):
            if child:
                counter[0] += 1
                cname = f"{name}-{counter[0]}"
                emit(child, cname)
                alts.append(f'"{_esc(label)}" {cname}' + ("?" if terminal else ""))
            else:
                alts.append(f'"{_esc(label)}"')
        rules.append(f"{rname} ::= " + " | ".join(alts))
    emit(tree, name)
    return rules


class SentenceGrammar:
    """One sentence of allowed words, numbers and light punctuation.

    The rules for the (large) base word list are built once; per sentence only
    the extra tokens (names and numbers of that sentence) are added."""

    def __init__(self, words):
        words = sorted({w for w in words if w})
        caps = sorted({w[:1].upper() + w[1:] for w in words})
        self.base = "\n".join(word_rules(words, "w") + word_rules(caps, "fw"))
        self.n_words = len(words)

    def text(self, extra_tokens=()) -> str:
        extra = sorted({t for t in extra_tokens if t and all(ch not in t for ch in '"\\\n')})
        rules = [self.base]
        alt = "w | fw"
        if extra:
            rules += word_rules(extra, "x")
            alt += " | x"
        # any allowed word, in lower case or capitalised, plus the sentence's own
        # names, numbers and signable words; numbers outside the sentence are not allowed
        rules.insert(0, f"root ::= ({alt}) (sep ({alt}))* end")
        rules.append('sep ::= " " | ", " | "; " | ": " | " - " | "\'s "')
        rules.append('end ::= "." | "?" | "!"')
        return "\n".join(rules) + "\n"
