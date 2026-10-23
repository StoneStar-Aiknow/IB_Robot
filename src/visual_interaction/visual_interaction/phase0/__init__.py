"""Phase-zero measurement harness. Nothing here runs in the interaction itself.

Each tool writes one ``measurement.json``. A measurement is evidence only for
the exact code, models, preprocessing and ABI it ran on; changing any of them
invalidates it (design §11), which is why every report carries its provenance.

The arithmetic lives in ROS-free modules that colcon tests on any host:
``stats``, ``report``, ``budget``, ``prompts``, ``pcm``, ``golden`` and
``tokenizer``. The command-line tools on top of them:

``token_budget``
    Counts this feature's prompts against the fixed OM shapes. Tokenizer data
    only — no model, no NPU, no CANN — so it runs locally.
``vl_segments``
    Summarises per-segment VL timings from a board demo run log.
``golden_compare``
    Compares two ``.npz`` archives of model outputs under tolerances frozen in
    a file beforehand.
``tts_timing``
    Times ``/voice_tts/synthesize`` under a labelled load condition. Synthesis
    only; it never plays.
``pcm_intervals``
    Listens to ``/audio/play`` and reports publish intervals and lateness
    against the sample clock. Passive; it never publishes.

None of these imports the board VL demo. Its licence is unconfirmed, so its
prompts, preprocessing and generation loop stay where they are: ``vl_segments``
reads the timings the demo already writes to its own log, and the token count
is built from the tokenizer data files, which are data rather than code.
"""
