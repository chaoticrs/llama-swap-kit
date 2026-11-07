# bin/espeak-ng/

eSpeak-ng 1.52.0 (from the official `espeak-ng.msi` release, unpacked with `msiexec /a`, nothing installed):
the text-to-phoneme library Kokoro needs for English, Hindi, Spanish, French, Italian and Portuguese.
1.52.0 is the version audio.cpp v0.9.3 is validated against.

- `libespeak-ng.dll`: the library. audio.cpp finds it through `AUDIOCPP_ESPEAK_LIBRARY` (set by `start.ps1`).
- `espeak-ng-data/`: pronunciation dictionaries and phoneme tables (441 files), found through `AUDIOCPP_ESPEAK_DATA`.
  Upstream data, not described file by file.

Licence: GPL-3.0-or-later (separate program loaded at runtime; not modified).
