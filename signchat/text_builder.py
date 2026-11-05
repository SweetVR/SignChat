"""Builds the chat message from typed letters."""

from __future__ import annotations

import re

CHATBOX_LIMIT = 144  # VRChat chatbox maximum


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


class TextBuilder:
    def __init__(self, case_mode: str = "sentence", max_len: int = CHATBOX_LIMIT):
        self.case_mode = case_mode
        self.max_len = max_len
        self.raw = ""

    def is_empty(self) -> bool:
        return not self.raw.strip()

    def add_letter(self, ch: str) -> None:
        self.raw += ch.upper()

    def add_space(self) -> bool:
        if self.raw and not self.raw.endswith(" "):
            self.raw += " "
            return True
        return False

    def backspace(self) -> str | None:
        if not self.raw:
            return None
        removed = self.raw[-1]
        self.raw = self.raw[:-1]
        return removed

    def last_char(self) -> str | None:
        return self.raw[-1] if self.raw else None

    def clear(self) -> None:
        self.raw = ""

    def display(self) -> str:
        return format_text(self.raw, self.case_mode)

    def pop_overflow(self) -> str | None:
        """If the message no longer fits in one chatbox, cut off a full first
        part (at a word boundary when possible) and return it for sending."""
        if len(self.raw.rstrip()) <= self.max_len:
            return None
        cut = self.raw.rfind(" ", 0, self.max_len + 1)
        if cut <= 0:
            cut = self.max_len
        head, tail = self.raw[:cut], self.raw[cut:].lstrip()
        self.raw = tail
        return format_text(head, self.case_mode).strip()

    def finalize(self) -> str:
        text = format_text(self.raw, self.case_mode).strip()
        self.raw = ""
        return text
