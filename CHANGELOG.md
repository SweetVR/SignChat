# Changelog

## 0.2.0

**Words and phrases**
- Teach SignChat whole signs: one- or two-handed, with movement. Start from a list of about 75
  common words and phrases (Add common signs…) or add your own, and record each one 3 times.
- Signs are recognised while you sign continuously (no pause needed between them), faster or
  slower than you recorded them.
- Every sign types its own text, so a phrase can type natural English: YOUR-NAME-WHAT →
  "what's your name?". A phrase recorded in one go replaces the separate signs it's made of.
- A sign takes back letters that were typed while you were signing it; fingerspelling and
  signs mix in one message ("Hello sam nice to meet you").
- Accuracy check covers words and phrases too.

**Fix a mistake**
- "Typed recently" shows the last letters and words. Click one to correct it; SignChat saves it
  as a new recording (or makes a word stricter if it was typed by mistake), so recognition
  improves while you chat.

**Template packs**
- Use someone else's recordings until you record your own, make a pack from yours to share,
  and download community packs (list in `packs/index.json`). Your own recordings always win.

**TTS Voice Wizard**
- Optional: finished messages go to TTS Voice Wizard's OSC listener (TextToSpeech or
  TextToText), which speaks, translates and/or shows them, instead of both apps overwriting
  the chatbox.

**Other**
- Hand positions are measured in a level frame that follows where you face but not where you
  look, so looking down at your hands no longer counts as raising them.
- Both hands are passed to the recogniser; the simulator signs words with both hands.
- The window title, buttons and VR panel mention signs, not only letters ("Delete last"
  removes the last letter or word).
- Faster feature extraction (vectorised).
- `.bat` files are checked out with Windows line endings.

## 0.1.0

- First version: ASL fingerspelling (26 letters, J and Z by their movement) to the VRChat
  chatbox over OSC, per-user recordings with guided setup, accuracy check, SteamVR overlay
  panel and haptics, webcam and simulator inputs.
