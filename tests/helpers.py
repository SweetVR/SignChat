"""Shared test utilities: a fake VRChat (captures OSC) and a simulated signer
that follows the engine's recording prompts."""

from __future__ import annotations

from signchat.config import Settings
from signchat.engine import Engine
from signchat.osc_out import ChatboxSender
from signchat.profile import Profile
from signchat.synth import REST_POSITION, SIGN_POSITION, Performer, SimUser


class OscCapture:
    def __init__(self):
        self.messages: list[tuple[str, list]] = []

    def __call__(self, address, args):
        self.messages.append((address, list(args)))

    def chat(self, final_only=False):
        out = [a for addr, a in self.messages if addr == "/chatbox/input"]
        return out

    def texts(self):
        return [a[0] for a in self.chat()]


def make_engine(settings: Settings | None = None, profile: Profile | None = None):
    settings = settings or Settings(source="sim")
    cap = OscCapture()
    events = []
    sender = ChatboxSender(min_interval=settings.live_interval_s, send_fn=cap)
    eng = Engine(settings, profile or Profile("test"), sender, events.append)
    eng.set_source_kind(settings.source)
    return eng, cap, events


def feed(engine: Engine, frames):
    """frames: HandFrames (one hand) or lists of HandFrames (both hands, as Performer.batches() gives)."""
    for f in frames:
        batch = f if isinstance(f, list) else [f]
        engine.on_update(batch, batch[0].t)


def feed_perf(engine: Engine, perf: Performer, start: int = 0) -> int:
    """Send everything the performer did since `start` (both hands). Returns the new start."""
    feed(engine, perf.batches(start))
    return len(perf.frames)


def drive_recording(engine: Engine, user: SimUser, seed: int = 7, t0: float = 0.0, max_s: float = 2000.0):
    """Act like a person following the recording prompts until the queue is empty."""
    perf = Performer(user, seed=seed, t0=t0)
    current = None
    motion_done_for = None
    t_end = t0 + max_s
    while engine.recording and perf.t < t_end:
        job = engine.job
        start = len(perf.frames)
        if job is None:
            perf._hold(0.05)
        elif engine.phase in ("prompt", "countdown"):
            if current != (job.label, job.take_no):
                current = (job.label, job.take_no)
                if job.kind == "sign":
                    perf.lower_hands(0.3)
                else:
                    pos = REST_POSITION if job.label == "_rest" else SIGN_POSITION
                    perf._move(user.take_pose(job.label, perf.rng), pos, 0.3)
            else:
                perf._hold(0.05)
        elif engine.phase == "capture":
            if job.kind == "sign" and motion_done_for != (job.label, job.take_no):
                motion_done_for = (job.label, job.take_no)
                perf._hold(0.3)
                perf.sign(job.label, engine.entry(job.label).kind)
                perf._hold(0.25)
                perf.lower_hands(0.4)
            elif job.kind == "sign":
                perf._hold(0.05)
            elif job.kind == "motion" and motion_done_for != (job.label, job.take_no):
                motion_done_for = (job.label, job.take_no)
                perf._hold(0.1)
                perf.motion(job.label)
                perf._hold(0.3)
            elif job.label == "_rest":
                drift = REST_POSITION + perf.rng.normal(0, 0.03, 3)
                perf._move(user.take_pose("_rest", perf.rng), drift, 0.25)
            else:
                perf._hold(0.05)
        feed(engine, perf.batches(start))
    return perf.t


_CACHE: dict = {}


def trained_engine(user: SimUser | None = None, settings: Settings | None = None, seed: int = 7,
                   cached: bool = True):
    """An engine whose profile was recorded through the guided flow by a simulated signer.
    Recordings are cached per signer so tests stay quick; the engine itself is always new."""
    user = user or SimUser(seed=11)
    key = (user.seed, user.hand, user.habit, user.take_var, user.jitter_m, seed)
    if cached and key in _CACHE:
        takes, t = _CACHE[key]
        prof = Profile("test")
        prof.takes = list(takes)
        eng, cap, events = make_engine(settings, prof)
        return eng, cap, events, user, t
    eng, cap, events = make_engine(settings)
    n = eng.guided()
    assert n > 0
    t = drive_recording(eng, user, seed=seed)
    _CACHE[key] = (list(eng.profile.takes), t)
    return eng, cap, events, user, t
