from signchat.segmenter import LetterSegmenter, TimingConfig


def run(seg, items, fps=60.0, t0=0.0):
    """items: list of (label, seconds, speed, active)."""
    t = t0
    events = []
    for label, dur, speed, active in items:
        for _ in range(int(round(dur * fps))):
            events += seg.update(t, label, 1.0 if label else 0.0, speed, active)
            t += 1 / fps
    return events, t


def kinds(events):
    return [(e.kind, e.value) for e in events]


def test_holding_a_shape_types_it_once():
    seg = LetterSegmenter(TimingConfig(word_gap_s=0, drop_send_s=0))
    ev, _ = run(seg, [("A", 1.0, 0.1, True)])
    assert kinds(ev) == [("letter", "A")]


def test_moving_hand_does_not_type():
    seg = LetterSegmenter(TimingConfig(word_gap_s=0, drop_send_s=0))
    ev, _ = run(seg, [("A", 1.0, 3.0, True)])
    assert ev == []


def test_double_letter_needs_a_bounce_or_a_different_shape():
    seg = LetterSegmenter(TimingConfig(word_gap_s=0, drop_send_s=0))
    ev, _ = run(seg, [("L", 0.6, 0.1, True), ("L", 0.25, 3.0, True), ("L", 0.6, 0.1, True),
                      (None, 0.3, 2.0, True), ("O", 0.6, 0.1, True)])
    assert kinds(ev) == [("letter", "L"), ("letter", "L"), ("letter", "O")]


def test_pause_adds_space_and_lowering_hand_sends():
    seg = LetterSegmenter(TimingConfig(word_gap_s=1.0, drop_send_s=0.8))
    ev, _ = run(seg, [("H", 0.6, 0.1, True), ("I", 1.6, 0.1, True), (None, 1.2, 0.0, False)])
    assert kinds(ev) == [("letter", "H"), ("letter", "I"), ("space", None), ("send", None)]


def test_action_signs_need_a_longer_hold():
    seg = LetterSegmenter(TimingConfig(word_gap_s=0, drop_send_s=0))
    ev, _ = run(seg, [("_backspace", 0.45, 0.1, True)])
    assert ev == []
    ev, _ = run(seg, [("_backspace", 0.3, 0.1, True)], t0=0.45)
    assert kinds(ev) == [("backspace", None)]


def test_rest_shape_never_types():
    seg = LetterSegmenter(TimingConfig(word_gap_s=0, drop_send_s=0))
    ev, _ = run(seg, [("_rest", 2.0, 0.0, True)])
    assert ev == []
