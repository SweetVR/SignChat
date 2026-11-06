"""Words and phrases through the whole app: recording prompts, live signing
mixed with fingerspelling, template packs, TTS Voice Wizard and fixing mistakes."""

from signchat import packs
from signchat.config import Settings
from signchat.engine import Engine
from signchat.osc_out import TTSVW_SPEAK, ChatboxSender, ttsvw_sink
from signchat.profile import Profile
from signchat.synth import Performer, SimUser
from tests.helpers import OscCapture, drive_recording, feed_perf, trained_engine

SIGNS = ["HELLO", "NICE", "MEET", "YOU", "FRIEND", "THANK-YOU", "NICE-MEET-YOU"]
_CACHE: dict = {}


def fast_settings(**kw) -> Settings:
    return Settings(source="sim", countdown_s=1.0, **kw)


def signing_engine(user: SimUser | None = None, settings: Settings | None = None, cached: bool = True):
    """An engine with letters and SIGNS recorded through the guided prompts by a simulated signer."""
    user = user or SimUser(seed=11)
    settings = settings or fast_settings(hand=user.hand)
    eng, cap, events, user, t = trained_engine(user, settings)
    key = (user.seed, user.hand)
    if cached and key in _CACHE:
        takes, meta, t = _CACHE[key]
        eng.profile.takes = list(takes)
        eng.profile.meta = {k: (dict(v) if isinstance(v, dict) else v) for k, v in meta.items()}
        eng.vocab = eng._load_vocab()
        eng.retrain()
        return eng, cap, events, user, t
    assert eng.add_common_signs(SIGNS) == len(SIGNS)
    n = eng.guided_signs()
    assert n == 3 * len(SIGNS)
    t = drive_recording(eng, user, seed=17, t0=t + 1)
    _CACHE[key] = (list(eng.profile.takes), {k: (dict(v) if isinstance(v, dict) else v)
                                             for k, v in eng.profile.meta.items()}, t)
    return eng, cap, events, user, t


def perform(eng, user, script, t0, seed=21, fps=60.0):
    perf = Performer(user, seed=seed, t0=t0, fps=fps)
    perf._hold(0.3)
    for item in script:
        if item.startswith("spell:"):
            perf.play(list(item[6:]))
        elif item.startswith("hold:"):
            perf._hold(float(item[5:]))
        elif item == "|":
            perf.lower_hands(0.4)
            perf._hold(1.6)
        else:
            perf.sign(item)
    perf.lower_hands(0.4)
    perf._hold(1.6)
    feed_perf(eng, perf)
    for i in range(1, 100):            # let the chatbox rate limit flush
        eng.on_update([], perf.t + i * 0.05)
    return perf.t + 6


def test_guided_sign_recording():
    eng, cap, events, user, t = signing_engine(cached=False)
    counts = eng.counts()
    for g in SIGNS:
        assert counts.get(g) == 3, g
    assert sorted(eng.bank.labels) == sorted(SIGNS)
    takes = [e for e in events if e["type"] == "take" and len(e["label"]) > 1]
    assert takes and all(e["ok"] for e in takes), [e["message"] for e in takes if not e["ok"]]
    phrase = [t for t in eng.profile.takes if t.label == "NICE-MEET-YOU"]
    assert all(t.points.shape[1:] == (2, 21, 3) and t.present.shape == (len(t.t), 2) for t in phrase)
    assert phrase[0].t[-1] - phrase[0].t[0] > 4.0          # phrases get the longer recording time


def test_signs_and_fingerspelling_in_one_message():
    eng, cap, events, user, t = signing_engine()
    eng.set_enabled(True)
    perform(eng, user, ["HELLO", "spell:SAM", "NICE", "MEET", "YOU"], t + 1)
    assert cap.texts()[-1] == "Hello sam nice to meet you", cap.texts()
    commits = [e for e in events if e["type"] == "commit" and e.get("sign")]
    assert [e["label"] for e in commits][-1] == "NICE-MEET-YOU"


def test_words_in_a_row_and_send_on_drop():
    eng, cap, events, user, t = signing_engine()
    eng.set_enabled(True)
    t = perform(eng, user, ["THANK-YOU", "FRIEND"], t + 1, seed=22)
    assert cap.texts()[-1] == "Thank you friend"
    perform(eng, user, ["NICE-MEET-YOU"], t, seed=23)
    assert cap.texts()[-1] == "Nice to meet you"


def test_holding_still_then_dropping_the_hands_adds_nothing():
    eng, cap, events, user, t = signing_engine()
    eng.set_enabled(True)
    for i, seed in enumerate((40, 49, 51)):
        t = perform(eng, user, ["NICE-MEET-YOU", "hold:1.5"], t + 1, seed=seed)
        assert cap.texts()[-1] == "Nice to meet you", (seed, cap.texts())


def test_left_handed_signer():
    user = SimUser(seed=12, hand="left")
    eng, cap, events, user, t = signing_engine(user, fast_settings(hand="left"))
    eng.set_enabled(True)
    perform(eng, user, ["HELLO", "FRIEND"], t + 1, seed=24)
    assert cap.texts()[-1] == "Hello friend"


def test_profile_with_signs_survives_restart():
    eng, *_ = signing_engine()
    eng.profile.save()
    loaded = Profile.load(eng.profile.path)
    assert loaded.meta["signs"]["NICE-MEET-YOU"]["text"] == "nice to meet you"
    sign_takes = [t for t in loaded.takes if t.kind == "sign"]
    assert len(sign_takes) == 3 * len(SIGNS) and all(t.present is not None for t in sign_takes)
    eng2 = Engine(fast_settings(), loaded, ChatboxSender(send_fn=OscCapture()))
    eng2.set_source_kind("sim")
    assert sorted(eng2.bank.labels) == sorted(SIGNS)
    assert eng2.entry("NICE-MEET-YOU").kind == "phrase"


def test_template_pack_from_another_signer(isolated_home):
    eng_a, *_ = signing_engine()
    path = packs.create_pack(eng_a.profile, SIGNS, "Test signs", author="A", tracking="simulated")
    info = packs.install_file(path)
    assert set(info.signs) == set(SIGNS) and info.letters == []
    assert [p.id for p in packs.list_installed()] == [info.id]

    user_b = SimUser(seed=13)
    eng_b, cap, events, user_b, t = trained_engine(user_b, fast_settings())   # B only recorded letters
    assert eng_b.bank is None
    eng_b.set_packs(packs.load_packs([info.id]))
    assert sorted(eng_b.bank.labels) == sorted(SIGNS)
    assert eng_b.pack_counts()["HELLO"] == 3
    rows = {r["gloss"]: r for r in eng_b.sign_rows()}
    assert rows["HELLO"]["pack"] == 3 and rows["HELLO"]["own"] == 0 and rows["HELLO"]["ready"]
    eng_b.set_enabled(True)
    perform(eng_b, user_b, ["HELLO", "spell:BO", "THANK-YOU"], t + 1, seed=25)
    assert cap.texts()[-1] == "Hello bo thank you", cap.texts()

    # recording a sign yourself replaces the pack's recordings of it
    eng_b.add_common_signs(["HELLO"])
    eng_b.guided_signs(["HELLO"])
    drive_recording(eng_b, user_b, seed=31, t0=t + 30)
    assert eng_b.pack_counts().get("HELLO", 0) == 0
    assert all(not tm.source for tm in eng_b.bank.templates if tm.label == "HELLO")


def test_tts_voice_wizard_gets_finished_messages():
    eng, cap, events, user, t = signing_engine()
    eng.s.live_captions = False
    tts = OscCapture()
    sender = ChatboxSender(min_interval=2.0, send_fn=cap, final_sink=ttsvw_sink(send_fn=tts))
    eng.set_sender(sender)
    eng.set_enabled(True)
    perform(eng, user, ["HELLO", "FRIEND"], t + 1, seed=26)
    assert tts.messages == [(TTSVW_SPEAK, ["Hello friend"])]
    assert cap.chat() == []                                  # SignChat didn't write the chatbox itself
    assert ("/chatbox/typing", [False]) in cap.messages


def test_fixing_mistakes_teaches_signchat():
    eng, cap, events, user, t = signing_engine()
    eng.seg.cfg.drop_send_s = 0.0                            # keep the message open
    eng.set_enabled(True)
    perform(eng, user, ["spell:AB", "FRIEND"], t + 1, seed=27)
    recent = eng.recent_tokens()
    assert [r["label"] for r in recent] == ["A", "B", "FRIEND"]
    assert eng.text.display() == "Ab friend"

    sign = recent[-1]
    msg = eng.fix_token(sign["id"], None)                   # "I wasn't signing that"
    assert "FRIEND" in msg and eng.sign_sensitivity("FRIEND") < 1.0
    assert eng.text.display() == "Ab"

    before = eng.counts().get("S", 0)
    msg = eng.fix_token(recent[1]["id"], "S")               # that B was really an S
    assert eng.counts()["S"] == before + 1 and "S" in msg
    assert eng.text.display() == "As"

    perform(eng, user, ["THANK-YOU"], t + 20, seed=28)
    tok = eng.recent_tokens()[-1]
    eng.fix_token(tok["id"], "HELLO")                       # it was HELLO: saved as a new recording
    assert eng.counts()["HELLO"] == 4 and eng.text.display().endswith("hello")
    assert any(tp.clip for tp in eng.profile.takes if tp.label == "HELLO")


def test_words_tab_rows_and_removing_a_sign():
    eng, *_ = signing_engine()
    rows = {r["gloss"]: r for r in eng.sign_rows()}
    assert rows["NICE-MEET-YOU"]["kind"] == "phrase" and rows["NICE-MEET-YOU"]["own"] == 3
    assert rows["MEET"]["two_handed"] in (True, False)
    err, g = eng.add_sign("good job", "good job!")
    assert err is None and g == "GOOD-JOB" and eng.entry("GOOD-JOB").text == "good job!"
    assert eng.add_sign("A")[0] is not None                 # single letters are fingerspelling
    assert eng.remove_sign("FRIEND") == 3
    assert "FRIEND" not in eng.bank.labels and "FRIEND" not in eng.vocab


def test_signs_are_ignored_while_paused_or_recording():
    eng, cap, events, user, t = signing_engine()
    perform(eng, user, ["HELLO", "FRIEND"], t + 1, seed=29)
    assert cap.texts() == []
    assert not [e for e in events if e["type"] == "commit"]
