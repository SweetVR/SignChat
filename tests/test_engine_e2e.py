"""Whole pipeline with a simulated signer: guided recording through the
engine's prompts, then fingerspelling into a fake VRChat."""

from signchat.config import Settings
from signchat.engine import GUIDED_ORDER, Engine
from signchat.osc_out import ChatboxSender
from signchat.profile import Profile
from signchat.synth import Performer, SimUser
from tests.helpers import OscCapture, feed, make_engine, trained_engine


def spell(eng, user, script, t0, seed=21):
    perf = Performer(user, seed=seed, t0=t0)
    frames = perf.play(script)
    feed(eng, frames)
    # let the rate limiter flush
    t = frames[-1].t
    for i in range(1, 400):
        eng.on_update([], t + i * 0.05)
    return t + 20


def final_messages(cap):
    return [a[0] for a in cap.chat()]


def test_guided_recording_covers_every_sign():
    eng, cap, events, user, t = trained_engine(cached=False)
    counts = eng.counts()
    for label in GUIDED_ORDER:
        want = 3 if label in ("J", "Z") else 1 if label == "_rest" else 2
        assert counts.get(label, 0) >= want, label
    assert set(eng.matcher.labels) == {"J", "Z"}
    assert not any(e["type"] == "take" and not e["ok"] for e in events)


def test_spelling_reaches_the_chatbox():
    eng, cap, events, user, t = trained_engine()
    eng.set_enabled(True)
    spell(eng, user, list("HELLO") + [" "] + list("JAZZ") + ["|"], t0=t + 1)
    msgs = final_messages(cap)
    assert msgs, events[-10:]
    assert msgs[-1] == "Hello jazz", msgs


def test_left_handed_signer():
    settings = Settings(source="sim", hand="left")
    eng, cap, events, user, t = trained_engine(SimUser(seed=12, hand="left"), settings)
    eng.set_enabled(True)
    spell(eng, user, list("WAVY") + ["|"], t0=t + 1)
    assert final_messages(cap)[-1] == "Wavy"


def test_nothing_is_typed_while_disabled_or_recording():
    eng, cap, events, user, t = trained_engine()
    spell(eng, user, list("ABC") + ["|"], t0=t + 1)
    assert final_messages(cap) == []


def test_profile_survives_restart(isolated_home):
    eng, cap, events, user, t = trained_engine()
    eng.profile.save()
    loaded = Profile.load(eng.profile.path)
    assert len(loaded.takes) == len(eng.profile.takes)
    eng2 = Engine(Settings(source="sim"), loaded, ChatboxSender(send_fn=OscCapture()))
    eng2.set_source_kind("sim")
    assert eng2.model.labels == eng.model.labels
    assert eng2.matcher.labels == ["J", "Z"]


def test_accuracy_report_runs():
    eng, *_ = trained_engine()
    rep = eng.evaluate()
    assert rep["static"].overall > 0.85
    for label in ("J", "Z"):
        hit, n = rep["motion"][label]
        assert n == 3 and hit >= 2


def test_recording_waits_for_a_hand_and_retries_a_failed_take():
    eng, cap, events = make_engine()
    eng.record("A", takes=1)
    t = 0.0
    for _ in range(300):              # 5 s without any hand: nothing starts, nothing fails
        eng.on_update([], t)
        t += 1 / 60
    assert eng.job is None and len(eng.jobs) == 1
    assert any(e["type"] == "rec" and e.get("waiting") for e in events)
    perf = Performer(SimUser(seed=3), t0=t)
    frames = perf.hold_sign("A", 2.5)  # hand appears: prompt + countdown start
    feed(eng, frames)
    assert eng.phase in ("countdown", "capture")
    t = frames[-1].t
    while eng.job is not None and t < 60:   # hand disappears before the capture ends
        t += 1 / 60
        eng.on_update([], t)
    fails = [e for e in events if e["type"] == "take"]
    assert len(fails) == 1 and not fails[0]["ok"] and "hand" in fails[0]["message"]
    assert len(eng.jobs) == 1            # queued again once


def test_noisy_fast_tracking_still_types_mostly_right():
    """3 mm tracking jitter at 90 Hz, sloppier signing than when recording."""
    user = SimUser(seed=99, habit=0.07, take_var=0.05, jitter_m=0.003)
    eng, cap, events, user, t = trained_engine(user)
    eng.set_enabled(True)
    sentences = ["HELLO WORLD", "NICE TO MEET YOU", "I AM DEAF", "BUZZ OFF"]
    chars_ok = chars = exact = 0
    for i, sentence in enumerate(sentences):
        frames = Performer(user, fps=90, seed=300 + i, t0=t + 1).play(list(sentence) + ["|"])
        feed(eng, frames)
        t = frames[-1].t
        for k in range(1, 80):
            eng.on_update([], t + k * 0.05)
        t += 4
        got = cap.texts()[-1].upper() if cap.texts() else ""
        exact += got == sentence
        chars += len(sentence)
        chars_ok += sum(a == b for a, b in zip(got, sentence)) if len(got) == len(sentence) else 0
    assert exact >= 3
    assert chars_ok / chars > 0.75
