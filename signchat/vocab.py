"""Words and phrases: what you call a sign (its gloss) and what it types.

A gloss is the usual way of writing a sign down: capital letters, with
hyphens joining the parts of a phrase (NICE-MEET-YOU).  Glosses need at least
two characters so they can't be confused with fingerspelled letters.

ASL isn't English word for word, so every sign has its own output text:
YOUR-NAME-WHAT types "what's your name?".  SignChat doesn't describe how to
form signs; look them up in an ASL dictionary (links below) or, better,
learn them from Deaf signers.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

GLOSS_RE = re.compile(r"^[A-Z0-9][A-Z0-9'\-]{1,39}$")
MAX_TEXT = 100
DICTIONARIES = (
    ("Lifeprint (ASLU)", "https://www.lifeprint.com/dictionary.htm"),
    ("Handspeak", "https://www.handspeak.com/word/"),
)


@dataclass
class VocabEntry:
    gloss: str
    text: str
    kind: str = "word"       # "word" or "phrase" (phrases get a longer recording time)
    hands: str = "auto"      # "auto", "one" or "two"
    category: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("gloss")
        return d

    @classmethod
    def from_dict(cls, gloss: str, d: dict) -> "VocabEntry":
        return cls(gloss=gloss, text=str(d.get("text") or default_text(gloss))[:MAX_TEXT],
                   kind="phrase" if d.get("kind") == "phrase" else "word",
                   hands=d.get("hands") if d.get("hands") in ("auto", "one", "two") else "auto",
                   category=str(d.get("category", "")))


def normalize_gloss(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def validate_gloss(g: str) -> str | None:
    """An error message, or None if the gloss is fine."""
    if len(g) < 2:
        return "Use at least 2 characters (single letters are fingerspelling)."
    if not GLOSS_RE.match(g):
        return "Use letters, numbers, - and ' only, for example NICE-MEET-YOU."
    return None


def default_text(gloss: str) -> str:
    return gloss.replace("-", " ").lower()


def _e(gloss, text, category, kind="word"):
    return VocabEntry(gloss, text, kind, "auto", category)


STARTER_VOCAB: tuple[VocabEntry, ...] = (
    # everyday words
    _e("HELLO", "hello", "Basics"), _e("BYE", "bye", "Basics"), _e("YES", "yes", "Basics"),
    _e("NO", "no", "Basics"), _e("PLEASE", "please", "Basics"), _e("THANK-YOU", "thank you", "Basics"),
    _e("SORRY", "sorry", "Basics"), _e("WELCOME", "welcome", "Basics"), _e("NICE", "nice", "Basics"),
    _e("MEET", "meet", "Basics"), _e("FRIEND", "friend", "Basics"), _e("GOOD", "good", "Basics"),
    _e("BAD", "bad", "Basics"), _e("FINE", "fine", "Basics"), _e("AGAIN", "again", "Basics"),
    _e("WAIT", "wait", "Basics"), _e("HELP", "help", "Basics"), _e("NOW", "now", "Basics"),
    _e("LATER", "later", "Basics"), _e("SEE", "see", "Basics"),
    # questions
    _e("WHAT", "what", "Questions"), _e("WHERE", "where", "Questions"), _e("WHO", "who", "Questions"),
    _e("WHY", "why", "Questions"), _e("HOW", "how", "Questions"), _e("WHEN", "when", "Questions"),
    _e("WHICH", "which", "Questions"),
    # people, feelings, chat
    _e("ME", "I", "People & chat"), _e("YOU", "you", "People & chat"), _e("MY", "my", "People & chat"),
    _e("YOUR", "your", "People & chat"), _e("NAME", "name", "People & chat"),
    _e("DEAF", "deaf", "People & chat"), _e("HEARING", "hearing", "People & chat"),
    _e("HARD-OF-HEARING", "hard of hearing", "People & chat"), _e("SIGN", "sign", "People & chat"),
    _e("ASL", "ASL", "People & chat"), _e("LEARN", "learn", "People & chat"),
    _e("UNDERSTAND", "understand", "People & chat"), _e("KNOW", "know", "People & chat"),
    _e("LIKE", "like", "People & chat"), _e("LOVE", "love", "People & chat"),
    _e("WANT", "want", "People & chat"), _e("NEED", "need", "People & chat"),
    _e("HAPPY", "happy", "People & chat"), _e("SAD", "sad", "People & chat"),
    _e("TIRED", "tired", "People & chat"), _e("FUNNY", "funny", "People & chat"),
    _e("COOL", "cool", "People & chat"), _e("SLOW", "slow", "People & chat"),
    _e("GO", "go", "People & chat"), _e("COME", "come", "People & chat"),
    _e("EAT", "eat", "People & chat"), _e("DRINK", "drink", "People & chat"),
    _e("SLEEP", "sleep", "People & chat"),
    _e("ILY", "I love you", "People & chat"),
    # whole phrases (recorded in one go)
    _e("NICE-MEET-YOU", "nice to meet you", "Phrases", "phrase"),
    _e("HOW-YOU", "how are you?", "Phrases", "phrase"),
    _e("YOUR-NAME-WHAT", "what's your name?", "Phrases", "phrase"),
    _e("MY-NAME", "my name is", "Phrases", "phrase"),
    _e("ME-DEAF", "I'm deaf", "Phrases", "phrase"),
    _e("ME-HARD-OF-HEARING", "I'm hard of hearing", "Phrases", "phrase"),
    _e("ME-HEARING", "I'm hearing", "Phrases", "phrase"),
    _e("ME-LEARN-SIGN", "I'm learning sign language", "Phrases", "phrase"),
    _e("YOU-DEAF", "are you deaf?", "Phrases", "phrase"),
    _e("YOU-UNDERSTAND", "do you understand?", "Phrases", "phrase"),
    _e("ME-NOT-UNDERSTAND", "I don't understand", "Phrases", "phrase"),
    _e("AGAIN-PLEASE", "can you repeat that?", "Phrases", "phrase"),
    _e("SIGN-SLOW-PLEASE", "please sign slower", "Phrases", "phrase"),
    _e("WHAT-UP", "what's up?", "Phrases", "phrase"),
    _e("SEE-YOU-LATER", "see you later", "Phrases", "phrase"),
    _e("GOOD-MORNING", "good morning", "Phrases", "phrase"),
    _e("GOOD-NIGHT", "good night", "Phrases", "phrase"),
    _e("ME-MUST-GO", "I have to go", "Phrases", "phrase"),
    _e("BRB", "brb", "Phrases", "phrase"),
)
STARTER_BY_GLOSS = {e.gloss: e for e in STARTER_VOCAB}
CATEGORIES = tuple(dict.fromkeys(e.category for e in STARTER_VOCAB))


def entry_for(gloss: str, vocab: dict[str, VocabEntry] | None = None) -> VocabEntry:
    if vocab and gloss in vocab:
        return vocab[gloss]
    if gloss in STARTER_BY_GLOSS:
        return STARTER_BY_GLOSS[gloss]
    return VocabEntry(gloss, default_text(gloss))
