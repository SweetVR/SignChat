"""The word/phrase recogniser on its own, with a simulated signer."""

import numpy as np
import pytest

from signchat import signs as sg
from signchat.synth import Performer, SimUser, sign_spec

WORDS = ["HELLO", "NICE", "MEET", "YOU", "FRIEND", "THANK-YOU", "WHAT", "SEE", "LATER", "PLEASE"]
PHRASES = ["NICE-MEET-YOU", "SEE-YOU-LATER"]


def pairs(perf):
    t, P, M = [], [], []
    for f, n in zip(perf.frames, perf.frames_n):
        pts = np.zeros((2, 21, 3), np.float32)
        m = np.zeros(2, bool)
        for fr in (f, n):
            if fr is not None:
                k = 0 if fr.hand == "right" else 1
                pts[k], m[k] = fr.level_points(), True
        t.append(f.t)
        P.append(pts)
        M.append(m)
    return np.array(t), np.stack(P), np.stack(M)


def recording(user, gloss, seed, hands_up_first=False):
    perf = Performer(user, seed=seed)
    if hands_up_first:
        perf._move_both(perf.pose, np.array([0.12, -0.35, -0.35]), 0.0, perf.npose, perf.npos, 0.0, 0.3)
    perf._hold(0.4)
    perf.sign(gloss)
    perf._hold(0.3)
    perf.lower_hands(0.4)
    perf._hold(0.4)
    return pairs(perf)


_BANKS = {}


def bank_for(user):
    key = (user.seed, user.hand)
    if key not in _BANKS:
        templates = []
        for gi, g in enumerate(WORDS + PHRASES):
            for k in range(3):
                t, P, M = recording(user, g, 100 + 31 * gi + k, hands_up_first=(k == 1))
                tm = sg.build_template(g, f"{g}-{k}", t, P, M, user.hand)
                assert tm is not None, g
                templates.append(tm)
        sg.harmonize_hands(templates)
        _BANKS[key] = sg.SignBank(templates)
    return _BANKS[key]


def run(bank, user, script, seed, speed=1.0):
    perf = Performer(user, seed=seed)
    perf._hold(0.4)
    for item in script:
        if item.startswith("spell:"):
            perf.play(list(item[6:]))
        else:
            perf.sign(item, None, speed)
    perf.lower_hands(0.4)
    perf._hold(0.8)
    enc = sg.LiveEncoder(user.hand)
    spot = sg.SignSpotter(bank)
    dets = []
    for batch in perf.batches():
        r = next((f.level_points() for f in batch if f.hand == "right"), None)
        l = next((f.level_points() for f in batch if f.hand == "left"), None)
        for g, x in enc.update(batch[0].t, r, l):
            dets += spot.push(x, g)
    return dets + spot.flush()


def final(dets):
    out = []
    for d in dets:   # what the text ends up as: a covering phrase replaces what it overlaps
        while out and out[-1].end_t - max(out[-1].start_t, d.start_t) >= 0.5 * (out[-1].end_t - out[-1].start_t):
            out.pop()
        out.append(d)
    return [d.label for d in out]


def test_recording_is_trimmed_to_the_sign():
    user = SimUser(seed=5)
    t, P, M = recording(user, "HELLO", 1)
    tm = sg.build_template("HELLO", "x", t, P, M, "right")
    total = t[-1] - t[0]
    assert tm.duration < total - 1.0        # raising and lowering the hands are cut off
    assert not tm.cut_off and not tm.static
    assert np.all(tm.F[:, 1] > sg.SPACE_Y)  # the dominant hand is up for the whole template


def test_one_and_two_handed_signs_are_told_apart():
    bank = bank_for(SimUser(seed=5))
    for g in WORDS:
        two = {t.two_handed for t in bank.templates if t.label == g}
        assert two == {bool(sign_spec(g).two)}, g


def test_signs_in_a_row_are_found():
    user = SimUser(seed=5)
    bank = bank_for(user)
    dets = run(bank, user, ["HELLO", "FRIEND", "THANK-YOU", "PLEASE"], seed=901)
    assert final(dets) == ["HELLO", "FRIEND", "THANK-YOU", "PLEASE"]
    dets = run(bank, user, ["WHAT", "YOU", "SEE"], seed=902, speed=1.3)     # faster than recorded
    assert final(dets) == ["WHAT", "YOU", "SEE"]


def test_phrase_replaces_its_words():
    user = SimUser(seed=5)
    bank = bank_for(user)
    dets = run(bank, user, ["HELLO", "NICE", "MEET", "YOU"], seed=903)
    assert final(dets) == ["HELLO", "NICE-MEET-YOU"]
    dets = run(bank, user, ["SEE-YOU-LATER"], seed=904)
    assert final(dets) == ["SEE-YOU-LATER"]


def test_left_handed_signer_uses_mirrored_templates():
    user = SimUser(seed=6, hand="left")
    bank = bank_for(user)
    assert final(run(bank, user, ["NICE", "FRIEND"], seed=905)) == ["NICE", "FRIEND"]


def test_fingerspelling_and_random_movement_are_mostly_ignored():
    user = SimUser(seed=5)
    bank = bank_for(user)
    false = 0
    for i, word in enumerate(["HELLO", "BRIGHT", "QUICK", "VRCHAT", "MOXY"]):
        false += len(run(bank, user, ["spell:" + word], seed=950 + i))
    assert false <= 1
    rng = np.random.default_rng(3)
    perf = Performer(user, seed=960)
    for _ in range(25):
        pose = user.take_pose(str(rng.choice(list("ABCDEFGHIKLMNOPRSTUVWXY") + ["_rest"])), perf.rng)
        pos = np.array([rng.uniform(-0.05, 0.3), rng.uniform(-0.55, 0.0), rng.uniform(-0.45, -0.15)])
        perf._move(pose, pos, float(rng.uniform(0.2, 0.7)))
        perf._hold(float(rng.uniform(0.0, 0.4)))
    enc, spot, dets = sg.LiveEncoder("right"), sg.SignSpotter(bank), []
    for batch in perf.batches():
        r = next((f.level_points() for f in batch if f.hand == "right"), None)
        l = next((f.level_points() for f in batch if f.hand == "left"), None)
        for g, x in enc.update(batch[0].t, r, l):
            dets += spot.push(x, g)
    assert len(dets) <= 1


def test_accuracy_check_and_thresholds():
    bank = bank_for(SimUser(seed=5))
    for label in bank.labels:
        assert sg.THRESHOLD_MIN <= bank.label_thr[label] <= sg.THRESHOLD_MAX
    rep = sg.evaluate_bank(bank)
    assert rep.overall >= 0.9, rep.worst()


def test_bank_needs_templates():
    with pytest.raises(ValueError):
        sg.SignBank([])
