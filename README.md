# SignChat: ASL → VRChat chatbox

SignChat watches your hands through SteamVR (Valve Index Knuckles, Quest hand tracking through
Steam Link or Virtual Desktop, and other SteamVR hands) and types what you sign into the VRChat
chatbox above your avatar. It runs next to VRChat, the way Kikitan Translator does for speech, and
sends text to VRChat's OSC port (9000), or to TTS Voice Wizard so it can speak it.

**What it does**
- **Fingerspelling**: the 26 ASL letters, including the moving letters J and Z
- **Words and phrases you teach it**: one- and two-handed signs with movement, recognised while you
  sign continuously. Start from a list of about 75 common words and phrases or add your own.
  A phrase you record in one go (NICE-MEET-YOU) types natural English ("nice to meet you") and
  replaces the separate signs.
- **Mix both**: sign words and fingerspell names in the same message
- **Learns *you***: you record each letter or sign a few times, so it adapts to your hands, your
  hardware and the way you sign. **Fix a mistake** turns wrong guesses into new examples while you chat.
- **Template packs**: start with someone else's recordings and share yours
- **Works with TTS Voice Wizard**: finished messages go to TTS Voice Wizard, which speaks, translates
  and/or shows them, instead of the two apps overwriting each other's chatbox text
- A small status panel inside VR, controller buzzes while you record, live text in the chatbox

**What it doesn't do (yet)**
- It doesn't translate ASL grammar. Every sign types the text you gave it. Signed word by word,
  YOUR NAME WHAT comes out as "your name what" unless you teach the phrase YOUR-NAME-WHAT →
  "what's your name?".
- It doesn't know any signs until you (or a template pack) teach them, and it doesn't teach signing.
  Look signs up in an ASL dictionary ([Lifeprint](https://www.lifeprint.com/dictionary.htm),
  [Handspeak](https://www.handspeak.com/word/)) or, better, learn them from Deaf signers.
- Facial expressions and mouth movements, which are part of ASL grammar, aren't tracked.
- It can't read finger positions your hardware doesn't track. Index controllers sense finger
  curl but not every detail, so a few letters may need a controller-friendly handshape.

> Status: 0.2. Fingerspelling has been tried in VRChat by the first testers. Words and phrases are
> new in this version and so far tested with simulated hands only, so please report how they work
> for you.

---

## Quick start (Windows)

1. Install **Python 3.12** from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**.
   (Skip this if you got a ready-made `SignChat.exe`.)
2. Double-click **`Start SignChat.bat`**. The first start installs what SignChat needs (a minute or two).
3. In VRChat turn on OSC: **Action Menu → Options → OSC → Enabled**.
4. Start SteamVR. The top of the SignChat window says **Connected to SteamVR** and shows your
   tracking type (`full` for hand tracking, `partial` for Index controllers).
5. **Letters** tab → **Guided setup**, and follow the prompts. They show in the SignChat window and on
   the panel in VR:
   - **Rest** first: relax and move your hand naturally. This teaches SignChat what "not
     signing" looks like.
   - Each letter twice: form the handshape during the countdown and hold it still while it records.
   - **J** and **Z** three times: form the handshape, and right after **GO** do the movement once, then stop.
   - Controller buzzes: short = countdown, long = go, double = saved, very long = try again.
6. **Words & phrases** tab (optional, see below) → **Add common signs…** → **Record missing**.
7. Press **Check accuracy**. It tests every recording against the others and lists letters and words
   that get mixed up. Re-record those, or make the two signs more different.
8. Press **Start signing**.

Don't want to record everything? **Template packs** lets you use someone else's recordings until you
record your own.

## Words and phrases

1. **Words & phrases** tab → **Add common signs…** and tick the ones you use, or **Add a sign…**:
   - *Sign (gloss)*: a name in capitals, hyphens between the signs of a phrase (`THANK-YOU`,
     `YOUR-NAME-WHAT`). Single letters are fingerspelling, so glosses have at least 2 characters.
   - *Types*: what goes into the chatbox (`what's your name?`).
   - Tick *It's a phrase* for several signs in a row; phrases get a longer recording (5 s instead of 3 s).
2. **Record missing** records each one 3 times. For every recording: hands down → after **GO** sign it
   once at your normal speed → hands down again. Two-handed signs work; SignChat works out from your
   recordings whether a sign uses your other hand (you can set it with **Edit…**).
3. **Check accuracy**, then **Start signing**. A word is typed about a quarter of a second after you
   finish it.

While you sign:
- If a letter got typed while you were signing a word, the word takes it back.
- A phrase you recorded replaces the separate words it's made of: NICE, MEET, YOU → "nice to meet you".
- Fingerspell names between signs: HELLO, S-A-M → "Hello sam".
- Signing the same thing again right away needs a short pause or a small reset of your hands.

## Fix a mistake

Everything you sign shows up under **Typed recently**. Click something that's wrong and say what you
really signed:
- **Nothing**: SignChat takes it back (and puts back any letters a wrong word replaced). A word typed
  by mistake must be signed more exactly from then on. A letter typed by mistake becomes a "rest"
  example for that handshape.
- **The letter … / The word/phrase …**: it's corrected and saved as a new recording, so the letter or
  word gets better the more you use SignChat.

## How typing works

| You do | SignChat does |
|---|---|
| Hold a handshape still for about 0.3 s | Types the letter (the ring fills up while you hold) |
| Double letter (LL, OO…) | Small bounce or slide sideways, then hold again |
| J or Z | Do the movement. If the hold before it typed an "I", it's replaced by "J" |
| Sign a word or phrase you taught it | Types its text, and takes back letters typed while you signed it |
| Keep your hand still for about 1.2 s after a fingerspelled word | Adds a space |
| Lower your hands (or hide them) for 1 s | Sends the message |
| Optional signs (Space, Del, Send, Clear) | Record any handshape you don't use for letters on those tiles |

The window also has **Send now**, **Delete last** (the last letter or word) and **Clear** buttons, and a
list of sent messages. Timings, capital letters and more are in **Settings**.

VRChat blocks chatbox spam for a while if messages come too fast. SignChat updates the chatbox
at most every 2 seconds while you sign. Raise that in Settings if you ever see a timeout.

## With TTS Voice Wizard (or Kikitan)

[TTS Voice Wizard](https://github.com/VRCWizard/TTS-Voice-Wizard) can speak your messages out loud,
translate them and show them in the chatbox. To use it with SignChat:

1. In TTS Voice Wizard, turn on its **OSC Listener** (default port 4026; see its
   [OSC Listener guide](https://github.com/VRCWizard/TTS-Voice-Wizard/wiki/OSC-Listener)).
2. In SignChat: **Settings → TTS Voice Wizard → Send finished messages to TTS Voice Wizard**, and choose
   whether it should speak them or only show them.

Finished messages then go to TTS Voice Wizard instead of straight to the chatbox, and its own settings
(voice, translation, chatbox) decide what happens, so the two apps stop overwriting each other.
If you want only TTS Voice Wizard's text in the chatbox, turn off *Show text in the chatbox while I sign*.

Kikitan Translator only writes to the chatbox, so whichever app sends last wins. Run SignChat and
Kikitan at different times, or use TTS Voice Wizard for speech-to-text and sign-to-speech together.

## Template packs

A template pack is someone else's recordings of letters and/or words. **Template packs** (top of the
window) lets you install a pack from a file, download community packs, and make a pack from your own
recordings to share. Your own recordings always win: once you record a letter or word, the pack's
version isn't used. Packs work best with the same kind of tracking they were recorded with.
How to share a pack with everyone: [packs/README.md](packs/README.md).

## Tips for good recognition

- **Record the way you sign in VR**: same hand position (in front of your shoulder), same speed.
- **Hand tracking (Quest)**: keep your hand well in front of the headset cameras and avoid
  fingers hiding behind each other. Turning the palm slightly helps for M, N, T, E.
- **Index controllers**: you can't cross fingers or tuck the thumb, so pick distinct
  controller-friendly shapes for letters like R, M/N, S/T/A/E. VRChat's signing communities
  (e.g. Helping Hands) have conventions for this.
- More recordings help: add a third take to letters that get confused, and use **Fix a mistake**.
- **Words**: record them the way you sign them in conversation, at your normal speed and size. If two
  words get mixed up, record both a few more times. Signs that are just a held handshape without
  movement can be confused with fingerspelling; a small movement helps.
- **Two-handed signs**: keep your other hand down (or relaxed) for one-handed signs while recording,
  so SignChat can tell which signs need both hands.
- Recordings are per input type: SteamVR, webcam and the simulator each keep their own.
  You can export and import recordings from the **Profile** menu.

## Troubleshooting

**"Connected, but SteamVR isn't sending hand data"**
SignChat brings default SteamVR bindings for Index, Touch, Vive, WMR, Cosmos and Reverb
controllers. For other devices, such as Steam Link's "Hand Tracking Controller", add the binding
once:
SteamVR → Settings → Controllers → **Manage Controller Bindings** → pick **SignChat** →
choose your controller → **Custom** → **Edit** → **Skeleton** section: set *Left hand skeleton*
to the left hand skeleton and *Right hand skeleton* to the right, and *Left/Right hand pose*
to the raw hand poses → **Save**.

**The chatbox doesn't show anything**
- OSC must be enabled in VRChat (see Quick start). If you changed VRChat's OSC port with
  `--osc=`, set the same port in Settings.
- Check the Sent list in SignChat: if messages appear there, SignChat is sending them.

**Wrong letters get typed**
Run **Check accuracy**, re-record the confused letters, or lower **Tolerance** in Settings. Make sure
you recorded **Rest**.

**A word gets typed when I didn't sign it**
Click it under **Typed recently** → *Nothing*. That word then has to be signed more exactly. If it keeps
happening, record it again or lower *Words & phrases → Tolerance* in Settings.

**A word isn't recognised**
Record it a few more times the way you sign it in conversation, or raise *Tolerance* a little. Phrases
must be signed in one go, the way you recorded them.

**Nothing gets typed**
The hand must be still. Watch the ring: it should fill up. If the letter shown keeps changing between
two letters, they look alike to SignChat. Re-record them or make them more different. If it says
"hand lowered", raise your hand or change the height in Settings.

**Logs** are in `%APPDATA%\SignChat\logs\signchat.log` (Profile → Open SignChat data folder).

## Optional extras

- **Avatar toggle**: add a Bool parameter called `SignChat` to your avatar's expression
  parameters and a toggle in your menu, then enable *Let an avatar toggle turn signing on/off* in
  Settings. SignChat then listens on port 9001. Only one app can listen on 9001, so use an OSC
  router if another tool already does.
- **Webcam** (VRChat desktop mode, or trying at your desk): run `Install webcam add-on.bat`, then choose
  *Webcam* in Settings. The hand model (about 8 MB) downloads on first use.
- **Simulated hands**: choose *Simulated hand* in Settings to try the whole app without VR. Type letters
  in the box or pick a word, and the virtual hands sign it (the simulated words are made-up movements,
  not real ASL). During recording they follow the prompts by themselves.

## Privacy

Everything runs on your PC. SignChat only talks to VRChat (and TTS Voice Wizard, if you turn that on)
on your own computer (127.0.0.1). It only goes online when you ask it to: the one-time download of the
webcam hand model if you use the webcam, and the community pack list and packs you download.
Your recordings stay in `%APPDATA%\SignChat\profiles`; a pack you make contains only hand positions.

## Make a shareable .exe

- On your PC: double-click `build_exe.bat` and share the `dist\SignChat` folder as a zip. No Python is
  needed to run it.
- Or push the project to GitHub and run the **Build Windows app** workflow (Actions tab). It
  runs the tests, builds the app and gives you a downloadable zip.

The .exe doesn't include the webcam add-on (it would add about 150 MB). Run from Python for webcam use.

## For developers

```
signchat/
  app.py           desktop window (PySide6)
  engine.py        frames in → letters and signs → chatbox; recording flow; fixing mistakes
  features.py      94 numbers per hand: shape, contacts, joint angles, orientation
  classifier.py    letters: per-user nearest-neighbour model with rejection + accuracy check
  segmenter.py     when a shape becomes a letter (hold voting, double letters, spaces, send)
  motion.py        J/Z: fingertip path matching (DTW) + shape checks
  signs.py         words/phrases: two-hand features, streaming subsequence DTW (SPRING) over all
                   templates at once, per-sign thresholds, phrase-over-word preference
  vocab.py         glosses, what they type, the built-in list of common signs
  text_builder.py  the message as timed tokens (letters, signs, spaces); take-back; 144-character split
  packs.py         template packs: make, install, community list
  osc_out.py       VRChat chatbox over OSC with rate limiting; TTS Voice Wizard output
  sources/         steamvr.py (OpenVR overlay app), webcam.py (MediaPipe), simulator.py
  steamvr/         SteamVR action manifest + default bindings
  synth.py         synthetic hands and made-up signs for the simulator and tests
packs/             community pack list (index.json)
tests/             unit tests and end-to-end runs with simulated signers
```

```
pip install -r requirements.txt pytest
python -m pytest -q          # ~2 minutes
python -m signchat           # run the app
```

How word recognition works: each recording becomes a template of 86 numbers per frame at 30 fps
(both hands' position relative to your head, hand shape, orientation, where the hands are relative to
each other, and how the dominant hand is moving). While you sign, a streaming dynamic-time-warping
matcher compares the last few seconds with every template at once, allowing signs to be up to 2× faster
or 3× slower than recorded and their first and last frames to blend into neighbouring signs. Each sign's
threshold comes from how much your own recordings of it differ. With simulated signers, words signed in
a row are recognised close to 100% of the time with your own recordings and about 85% with someone
else's (a template pack); real signing will be messier, so reports are very welcome.

Ideas for later: practice mode, word suggestions, smoothing word-by-word glosses into English, numbers,
a chatbox relay so Kikitan and SignChat can share the chatbox, auto-start with SteamVR, OSCQuery.
