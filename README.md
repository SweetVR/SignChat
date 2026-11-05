# SignChat: ASL fingerspelling → VRChat chatbox

SignChat watches your hands through SteamVR (Valve Index Knuckles, Quest hand tracking through
Steam Link or Virtual Desktop, and other SteamVR hands) and types what you **fingerspell** into
the VRChat chatbox above your avatar. It works the way Kikitan Translator does for speech: it runs
next to VRChat and sends text to VRChat's OSC port (9000).

**What it does**
- Recognises the 26 ASL fingerspelling letters, including the moving letters J and Z
- Learns *your* handshapes: you record each letter a couple of times (about 3 minutes in total),
  so it adapts to your hand, your hardware and the handshapes you actually use in VR
- Shows a small status panel inside VR and buzzes your controllers while you record
- Shows letters in the chatbox while you spell (optional) and sends the message when you lower your hand

**What it doesn't do (yet)**
- It doesn't translate full ASL signs or grammar into English. Fingerspelling only.
- It can't read finger positions your hardware doesn't track. Index controllers sense finger
  curl but not every detail (crossed fingers, thumb tucked under fingers), so a few letters may
  need a controller-friendly handshape. The accuracy check tells you which ones.

> Status: early version (0.1). The recognition engine is tested with simulated hands. It has not
> been tested with real VR hardware yet, so please report what works and what doesn't.

---

## Quick start (Windows)

1. Install **Python 3.12** from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**.
   (Skip this if you got a ready-made `SignChat.exe`.)
2. Double-click **`Start SignChat.bat`**. The first start installs what SignChat needs (a minute or two).
3. In VRChat turn on OSC: **Action Menu → Options → OSC → Enabled**.
4. Start SteamVR. The top of the SignChat window says **Connected to SteamVR** and shows your
   tracking type (`full` for hand tracking, `partial` for Index controllers).
5. Press **Guided setup** and follow the prompts. They show in the SignChat window and on the panel
   in VR:
   - **Rest** first: relax and move your hand naturally. This teaches SignChat what "not
     signing" looks like.
   - Each letter twice: form the handshape during the countdown and hold it still while it records.
   - **J** and **Z** three times: form the handshape, and right after **GO** do the movement once, then stop.
   - Controller buzzes: short = countdown, long = go, double = saved, very long = try again.
6. Press **Check accuracy**. It tests every recording against the others and lists letters that get
   mixed up. Re-record those (select the tile → **Record**), or make the two handshapes more different.
7. Press **Start signing** and fingerspell.

## How typing works

| You do | SignChat does |
|---|---|
| Hold a handshape still for about 0.3 s | Types the letter (the ring fills up while you hold) |
| Double letter (LL, OO…) | Small bounce or slide sideways, then hold again |
| J or Z | Do the movement. If the hold before it typed an "I", it's replaced by "J" |
| Keep your hand still for about 1.2 s after a word | Adds a space |
| Lower your hand (or hide it) for 1 s | Sends the message |
| Optional signs (Space, Del, Send, Clear) | Record any handshape you don't use for letters on those tiles |

The window also has **Send now**, **Delete letter** and **Clear** buttons, and a list of sent
messages. Timings, capital letters and more are in **Settings**.

VRChat blocks chatbox spam for a while if messages come too fast. SignChat updates the chatbox
at most every 2 seconds while you spell. Raise that in Settings if you ever see a timeout.

## Tips for good recognition

- **Record the way you sign in VR**: same hand position (in front of your shoulder), same speed.
- **Hand tracking (Quest)**: keep your hand well in front of the headset cameras and avoid
  fingers hiding behind each other. Turning the palm slightly helps for M, N, T, E.
- **Index controllers**: you can't cross fingers or tuck the thumb, so pick distinct
  controller-friendly shapes for letters like R, M/N, S/T/A/E. VRChat's signing communities
  (e.g. Helping Hands) have conventions for this.
- More recordings help: add a third take to letters that get confused.
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
- **Simulated hand**: choose *Simulated hand* in Settings to try the whole app without VR. Type letters
  in the box and the virtual hand signs them. During Guided setup it follows the prompts by itself.

## Privacy

Everything runs on your PC. SignChat only talks to VRChat on your own computer (127.0.0.1). The one
exception is the one-time download of the webcam hand model, and only if you use the webcam.
Your recordings stay in `%APPDATA%\SignChat\profiles`.

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
  engine.py        frames in → letters → chatbox; recording flow
  features.py      94 numbers per frame: hand shape, contacts, joint angles, orientation
  classifier.py    per-user nearest-neighbour model with rejection + accuracy check
  segmenter.py     when a shape becomes a letter (hold voting, double letters, spaces, send)
  motion.py        J/Z: fingertip path matching (DTW) + shape checks
  text_builder.py  message assembly, sentence case, 144-character split
  osc_out.py       VRChat chatbox over OSC with rate limiting
  sources/         steamvr.py (OpenVR overlay app), webcam.py (MediaPipe), simulator.py
  steamvr/         SteamVR action manifest + default bindings
  synth.py         synthetic hands for the simulator and tests
tests/             unit tests and end-to-end runs with simulated signers
```

```
pip install -r requirements.txt pytest
python -m pytest -q          # ~1 minute
python -m signchat           # run the app
```

Ideas for later: word suggestions to fix single wrong letters, numbers, whole-word signs (the
motion matcher already handles any recorded movement), auto-start with SteamVR, OSCQuery so it can
share port 9001, sharing community profiles.
