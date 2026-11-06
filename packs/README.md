# Community template packs

A template pack is a set of recordings (letters and/or words and phrases) that other
people can use in SignChat until they record their own. SignChat lists the packs in
[`index.json`](index.json) under **Template packs → Community packs…**.

## Make a pack

1. Record letters and words in SignChat the way you normally sign them. Run
   **Check accuracy** first; only share recordings that work well for you.
2. **Template packs → Make a pack from my recordings…** Give it a name, say what you
   recorded it with (Valve Index controllers, Quest hand tracking, …) and save the file.
3. Try it: install the file on a second profile (or ask a friend) and check that it works
   for someone else too.

A pack contains only hand positions (numbers). No audio, video, names or anything else
from your PC. It's a `.npz` file that SignChat opens with pickling disabled.

## Add it to the list

1. Upload the `.npz` somewhere with a stable **https** download link, for example as a
   file in a GitHub release of your own repository.
2. Open a pull request that adds an entry to `index.json`:

```json
{
  "id": "everyday-asl-index",
  "name": "Everyday ASL (Index)",
  "author": "your name",
  "description": "About 60 common words and phrases, signed right-handed.",
  "tracking": "Valve Index controllers",
  "contents": "26 letters, 60 words/phrases",
  "size": 4200000,
  "url": "https://github.com/you/your-repo/releases/download/v1/everyday-asl-index.npz"
}
```

Please only share signs as they are actually used by Deaf signers, and say in the
description if a pack uses controller-friendly handshapes.

## How packs are used

- Your own recordings always win: as soon as you record a letter or a word yourself, the
  pack's recordings of it are no longer used.
- Packs are matched a little more loosely than your own recordings, because everyone
  signs a bit differently.
- They work best with the same kind of tracking they were recorded with. Index
  controllers and camera-based hand tracking see fingers quite differently.
