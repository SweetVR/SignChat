"""Builds the chat message from typed letters and recognised signs.

The message is a list of tokens (a fingerspelled letter, a whole sign, or a
space), each with the time span it was signed in.  That lets a sign that is
recognised a moment later take back the letters typed while it was being
signed, and a phrase replace the separate signs it is made of.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import count

CHATBOX_LIMIT = 144  # VRChat chatbox maximum
_ids = count(1)


def format_text(raw: str, case_mode: str) -> str:
    if case_mode == "upper":
        return raw.upper()
    text = raw.lower()
    if case_mode == "lower":
        return text
    # sentence case: capital first letter, and a lone "i" becomes "I"
    text = re.sub(r"\bi\b", "I", text)
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


@dataclass
class Token:
    kind: str                 # "letter", "word" (a recognised sign) or "space"
    text: str
    t0: float = 0.0
    t1: float = 0.0
    label: str = ""           # what was recognised: the letter, or the sign's gloss
    id: int = field(default_factory=lambda: next(_ids))


def _capitalize_sentences(text: str) -> str:
    out, start = [], True
    for ch in text:
        if start and ch.isalpha():
            out.append(ch.upper())
            start = False
            continue
        out.append(ch)
        if ch in ".!?":
            start = True
        elif not ch.isspace() and ch.isalnum():
            start = False
    return "".join(out)


class TextBuilder:
    def __init__(self, case_mode: str = "sentence", max_len: int = CHATBOX_LIMIT):
        self.case_mode = case_mode
        self.max_len = max_len
        self.tokens: list[Token] = []

    # ------------------------------------------------------------ editing
    def is_empty(self) -> bool:
        return not any(t.kind != "space" for t in self.tokens)

    def add_letter(self, ch: str, t0: float = 0.0, t1: float = 0.0) -> Token:
        tok = Token("letter", ch.upper(), t0, t1, ch.upper())
        self.tokens.append(tok)
        return tok

    def add_space(self, t: float = 0.0) -> bool:
        if not self.tokens or self.tokens[-1].kind != "letter":
            return False      # nothing yet, already a space, or right after a sign (spaced anyway)
        self.tokens.append(Token("space", " ", t, t))
        return True

    def retract_since(self, t0: float, letter_pad: float = 0.05) -> list[Token]:
        """Take back what was typed while a sign was being signed (from t0):
        letters whose hold started after t0 - letter_pad, spaces after that,
        and signs that mostly overlap."""
        removed: list[Token] = []
        while self.tokens:
            tok = self.tokens[-1]
            if tok.kind == "word":
                dur = max(tok.t1 - tok.t0, 1e-3)
                if tok.t1 - max(tok.t0, t0) < 0.5 * dur:
                    break
            elif tok.kind == "letter":
                if tok.t0 < t0 - letter_pad:
                    break
            elif tok.t1 < t0 - letter_pad:
                break
            removed.append(self.tokens.pop())
        removed.reverse()
        return removed

    def letters_since(self, t0: float, t1: float | None = None, letter_pad: float = 0.05) -> int:
        """How many fingerspelled letters a sign signed from t0 to t1 would take back."""
        n = 0
        for tok in reversed(self.tokens):
            if tok.kind == "word":
                break
            if tok.kind == "letter":
                if t1 is not None and tok.t0 > t1 + 0.05:
                    continue
                if tok.t0 < t0 - letter_pad:
                    break
                n += 1
        return n

    def add_word(self, text: str, label: str = "", t0: float = 0.0, t1: float = 0.0,
                 retract: bool = True) -> tuple[Token, list[Token]]:
        """Add a recognised sign (signed from t0 to t1). Letters typed after it
        ended stay after it; what was typed while it was signed is taken back."""
        after: list[Token] = []
        while self.tokens and self.tokens[-1].kind != "word" and self.tokens[-1].t0 > t1 + 0.05:
            after.append(self.tokens.pop())
        removed = self.retract_since(t0) if retract else []
        tok = Token("word", text, t0, t1, label)
        self.tokens.append(tok)
        self.tokens.extend(reversed(after))
        return tok, removed

    def backspace(self) -> str | None:
        """Remove the last letter, sign or space."""
        if not self.tokens:
            return None
        return self.tokens.pop().text

    def last_char(self) -> str | None:
        """The last fingerspelled letter, if the message ends with one."""
        return self.tokens[-1].text if self.tokens and self.tokens[-1].kind == "letter" else None

    def clear(self) -> None:
        self.tokens = []

    def find(self, token_id: int) -> Token | None:
        return next((t for t in self.tokens if t.id == token_id), None)

    def replace(self, token_id: int, kind: str, text: str, label: str) -> bool:
        tok = self.find(token_id)
        if tok is None:
            return False
        tok.kind, tok.text, tok.label = kind, text, label
        return True

    def replace_with(self, token_id: int, tokens: list[Token]) -> bool:
        """Put `tokens` where token_id was (e.g. the letters a wrong sign had taken back)."""
        for i, t in enumerate(self.tokens):
            if t.id == token_id:
                self.tokens[i:i + 1] = list(tokens)
                return True
        return False

    def remove(self, token_id: int) -> bool:
        before = len(self.tokens)
        self.tokens = [t for t in self.tokens if t.id != token_id]
        return len(self.tokens) != before

    # ------------------------------------------------------------ text
    def _pieces(self, tokens: list[Token]) -> list[tuple[int | None, str]]:
        """(token index or None for an automatic space, text) in order."""
        out: list[tuple[int | None, str]] = []
        prev: Token | None = None
        for idx, tok in enumerate(tokens):
            if tok.kind == "space":
                if out and not out[-1][1].endswith(" "):
                    out.append((idx, " "))
            else:
                needs_gap = prev is not None and prev.kind != "space" and (tok.kind == "word" or prev.kind == "word")
                if needs_gap and out and not out[-1][1].endswith(" "):
                    out.append((None, " "))
                out.append((idx, tok.text))
            prev = tok
        return out

    def _render(self, tokens: list[Token], case_mode: str | None = None) -> str:
        mode = case_mode or self.case_mode
        parts = []
        for idx, text in self._pieces(tokens):
            tok = tokens[idx] if idx is not None else None
            if mode == "upper":
                parts.append(text.upper())
            elif mode == "lower" or (tok is not None and tok.kind == "letter"):
                parts.append(text.lower())
            else:
                parts.append(text)
        text = "".join(parts)
        if mode == "sentence":
            text = re.sub(r"\bi\b", "I", text)
            text = _capitalize_sentences(text)
        return text

    @property
    def raw(self) -> str:
        return "".join(p for _, p in self._pieces(self.tokens))

    @raw.setter
    def raw(self, value: str) -> None:
        self.tokens = []
        for ch in value:
            if ch == " ":
                self.tokens.append(Token("space", " "))
            else:
                self.tokens.append(Token("letter", ch.upper(), label=ch.upper()))

    def display(self) -> str:
        return self._render(self.tokens)

    def pop_overflow(self) -> str | None:
        """If the message no longer fits in one chatbox, cut off a full first
        part (at a word boundary when possible) and return it for sending."""
        pieces = self._pieces(self.tokens)
        raw = "".join(p for _, p in pieces)
        if len(raw.rstrip()) <= self.max_len:
            return None
        starts, pos = [], 0
        for _, p in pieces:
            starts.append(pos)
            pos += len(p)
        cut = None
        for i, (_, p) in enumerate(pieces):           # last space that keeps the first part short enough
            if p == " " and 0 < starts[i] <= self.max_len:
                cut = i
        if cut is None:                                # one very long word: cut between letters
            fits = [i for i in range(len(pieces)) if starts[i] + len(pieces[i][1]) <= self.max_len]
            cut = fits[-1] + 1 if fits else 1
        head_idx = [idx for idx, _ in pieces[:cut] if idx is not None]
        k = (max(head_idx) + 1) if head_idx else 1
        head, tail = self.tokens[:k], self.tokens[k:]
        while tail and tail[0].kind == "space":
            tail.pop(0)
        self.tokens = tail
        return self._render(head).strip()

    def finalize(self) -> str:
        text = self.display().strip()
        self.tokens = []
        return text

    def recent(self, n: int = 12) -> list[Token]:
        return [t for t in self.tokens if t.kind != "space"][-n:]
