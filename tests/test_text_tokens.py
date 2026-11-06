from signchat.text_builder import TextBuilder


def spell(tb, word, t):
    for ch in word:
        tb.add_letter(ch, t, t + 0.3)
        t += 0.6
    return t


def test_signs_and_letters_are_spaced_and_cased():
    tb = TextBuilder()
    tb.add_word("hello", "HELLO", 0.0, 0.8)
    t = spell(tb, "SAM", 1.0)
    tb.add_word("nice to meet you", "NICE-MEET-YOU", t, t + 2)
    assert tb.display() == "Hello sam nice to meet you"
    tb.add_word("what's your name?", "YOUR-NAME-WHAT", t + 3, t + 5)
    tb.add_word("I'm deaf", "ME-DEAF", t + 6, t + 7)
    assert tb.display().endswith("you what's your name? I'm deaf")
    tb.case_mode = "upper"
    assert tb.display().startswith("HELLO SAM")
    tb.case_mode = "lower"
    assert tb.display().endswith("i'm deaf")


def test_sign_takes_back_letters_typed_while_it_was_signed():
    tb = TextBuilder()
    spell(tb, "HI", 0.0)                 # letters at 0.0-0.3 and 0.6-0.9
    tb.add_space(1.5)
    tb.add_letter("Y", 3.0, 3.3)         # typed by accident at the start of a sign
    tok, removed = tb.add_word("you", "YOU", 3.0, 3.8)
    assert [t.text for t in removed] == ["Y"]
    assert tb.display() == "Hi you"


def test_phrase_replaces_the_words_it_contains():
    tb = TextBuilder()
    tb.add_word("hello", "HELLO", 0.0, 0.8)
    tb.add_word("nice", "NICE", 1.0, 1.8)
    tb.add_word("meet", "MEET", 2.0, 2.7)
    tb.add_word("you", "YOU", 2.9, 3.5)
    tok, removed = tb.add_word("nice to meet you", "NICE-MEET-YOU", 1.0, 3.5)
    assert [t.label for t in removed] == ["NICE", "MEET", "YOU"]
    assert tb.display() == "Hello nice to meet you"


def test_neighbouring_sign_that_barely_overlaps_is_kept():
    tb = TextBuilder()
    tb.add_word("hello", "HELLO", 0.0, 1.3)
    tb.add_word("friend", "FRIEND", 1.0, 1.8)
    assert tb.display() == "Hello friend"
    assert tb.letters_since(0.0) == 0


def test_backspace_removes_a_whole_sign_and_fixes_can_restore_letters():
    tb = TextBuilder()
    spell(tb, "OK", 0.0)
    letters = list(tb.tokens)
    tb.tokens = []
    tok = tb.add_word("thank you", "THANK-YOU", 0.0, 1.0)[0]
    assert tb.replace_with(tok.id, letters)
    assert tb.display() == "Ok"
    tb.add_word("bye", "BYE", 2.0, 3.0)
    assert tb.backspace() == "bye"
    assert tb.display() == "Ok"


def test_long_message_with_signs_splits_between_tokens():
    tb = TextBuilder(max_len=40)
    t = 0.0
    for _ in range(6):
        tb.add_word("nice to meet you", "NICE-MEET-YOU", t, t + 1)
        t += 2
    first = tb.pop_overflow()
    assert first == "Nice to meet you nice to meet you"
    assert tb.display().startswith("Nice to meet you")
