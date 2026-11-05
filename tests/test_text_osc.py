import threading
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from signchat.osc_out import ChatboxSender
from signchat.text_builder import TextBuilder, format_text
from tests.helpers import OscCapture


def test_sentence_case_and_lone_i():
    assert format_text("HELLO I AM HERE", "sentence") == "Hello I am here"
    assert format_text("HI", "upper") == "HI"
    assert format_text("HI", "lower") == "hi"


def test_text_builder_spaces_and_backspace():
    tb = TextBuilder()
    for ch in "HI":
        tb.add_letter(ch)
    assert tb.add_space() is True
    assert tb.add_space() is False  # never two spaces
    tb.add_letter("X")
    assert tb.backspace() == "X"
    assert tb.finalize() == "Hi"
    assert tb.is_empty()


def test_long_messages_split_at_a_word_boundary():
    tb = TextBuilder(max_len=144)
    words = ["WORD"] * 40
    tb.raw = " ".join(words)
    first = tb.pop_overflow()
    assert first is not None and len(first) <= 144 and not first.endswith(" ")
    assert first.split()[-1].lower() == "word"
    assert len(tb.raw) < 144 and tb.pop_overflow() is None


def test_sender_rate_limits_and_keeps_final_messages():
    cap = OscCapture()
    s = ChatboxSender(min_interval=2.0, send_fn=cap)
    s.set_typing(True)
    s.show_live("Hel")
    s.tick(0.0)
    s.show_live("Hell")
    s.tick(0.5)          # too soon
    s.show_live("Hello")
    s.send_final("Hello there")
    s.show_live("Ne")    # next message starts before the final went out
    s.tick(1.0)          # still too soon
    s.tick(2.1)          # final goes first
    s.tick(4.2)          # then the new live text
    chat = cap.chat()
    assert [c[0] for c in chat] == ["Hel", "Hello there", "Ne"]
    assert chat[1][1] is True and chat[1][2] is False   # send now, no sound
    assert ("/chatbox/typing", [True]) in cap.messages
    assert ("/chatbox/typing", [False]) in cap.messages


def test_real_udp_messages_match_vrchat_format():
    received = []
    disp = Dispatcher()
    disp.map("/chatbox/*", lambda addr, *args: received.append((addr, args)))
    server = ThreadingOSCUDPServer(("127.0.0.1", 0), disp)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        s = ChatboxSender(port=port, min_interval=0.0)
        s.set_typing(True)
        s.send_final("Hello world")
        s.tick(0.0)
        deadline = time.time() + 2
        while len(received) < 3 and time.time() < deadline:
            time.sleep(0.02)
    finally:
        server.shutdown()
    assert ("/chatbox/typing", (True,)) in received
    assert ("/chatbox/input", ("Hello world", True, False)) in received
    assert ("/chatbox/typing", (False,)) in received
