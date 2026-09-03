"""Recording tooling: ancillary utilities that operate on ``.cast`` artifacts.

Currently ships :mod:`extensions.recording.tools.cast_to_mp4`, which
turns a recorded ``.cast`` file into a sequence of PNG frames and then
encodes the sequence into an MP4 (or any other ffmpeg-supported video
container).

These utilities are *opt-in conversion helpers* — they are not part of
the recording pipeline itself and do **not** need to be installed for
``clawcodex record`` to function. Users only pay for the dependency
(Pillow + ffmpeg) when they explicitly invoke a cast-to-* conversion.
"""
