"""The real vision-language port: live frames in, closed-schema answer out.

Three pieces have to meet here, and each belongs to someone else:

* **Decode** is ``manipulation_execution.frame_decoder``, built by name from
  configuration. PR 504 measured the pure-CPU path at ~90 ms per frame on the
  head stream, which cannot keep 30 Hz, so the default is ``auto`` (310P DVPP
  for JPEG, CPU for PNG). This module never decodes a JPEG itself.
* **Preprocessing and generation** are the board's InternVL deployment at
  ``demo_root``. Its preprocessing (BGR→RGB, bicubic 448², ImageNet
  normalisation), its chat-template token ids, its prefill/decode loop with the
  fixed-length KV cache — all of it is loaded from there rather than
  reimplemented. A reimplementation would have to be proved equivalent before
  it could be trusted, and nothing here needs it to be different.
* **The prompt and the answer's meaning** are ours:
  ``visual_interaction.phase0.prompts`` builds the closed-schema question and
  ``visual_interaction.session.semantics`` decides what the answer is allowed
  to say. This port returns the model's raw text; the session parses it.

So this file is an adapter, deliberately thin. What it does own is the failure
policy: a model that cannot be reached returns ``None``, which the session
reports as MODEL_NOT_READY rather than as silence, and every other fault
(decode failure, a window with too few frames, a generation that raises) is
reported the same way rather than being turned into a fabricated answer.

Loading is lazy. Importing this module must not touch the NPU, so a test can
import it on a workstation; the OMs load on the first ``observe`` and are kept
for the process's life.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from visual_interaction.session.hooks import RoundInputs, VisionLanguagePort

# What the board's export was compiled with. Not configurable here: these are
# properties of the OM files, and phase0.budget checks prompts against them.
IMAGE_TOKENS = 256
PREFILL_LEN = 1280
CONTEXT_LEN = 1344
VISION_BATCH = 4

# The demo modules this adapter needs, and where they live under demo_root.
_SCRIPTS = "scripts"
_LIVE = "live"
_VISION_OM = "internvl3_1b_vision_b4_fp16"
_PREFILL_OM = "llm_prefill_fp16"
_DECODE_OM = "llm_decode_fp16"
_LLM_OUTPUTS = ["logits", "new_key", "new_value"]
_EMBED_TABLE = "onnx/embed_tokens_fp32.npy"


class VisionBackendError(RuntimeError):
    """The model could not be loaded or reached. Reported as MODEL_NOT_READY."""


@dataclass
class InternVlVision(VisionLanguagePort):
    """Answers one window by running the board's InternVL deployment.

    ``demo_root`` is the board's InternVL directory. ``frames`` must match what
    the prompt asks for; the vision OM is compiled for a batch of
    :data:`VISION_BATCH`, so a smaller window is zero-padded and only its own
    embeddings are spliced into the prompt.
    """

    demo_root: str
    frames: int = 2
    max_new_tokens: int = 110
    categories: tuple[str, ...] = ()
    # "observe" asks the closed-schema question; "chatter" the short open one
    # of visual_interaction.session.chatter, whose answer is a phrase.
    prompt_mode: str = "observe"
    # Set on failure so the node can report *why* the model is not ready
    # instead of only that it is not.
    last_error: str = field(default="", init=False)
    prompt_tokens: int = field(default=0, init=False)
    _loaded: bool = field(default=False, init=False)
    _tokenizer: object = field(default=None, init=False)
    _detokenizer: object = field(default=None, init=False)
    _vision: object = field(default=None, init=False)
    _generator: object = field(default=None, init=False)
    _prompt_ids: object = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.prompt_mode not in ("observe", "chatter"):
            raise ValueError(f"prompt_mode must be 'observe' or 'chatter', got {self.prompt_mode!r}")
        if self.frames < 1 or self.frames > VISION_BATCH:
            raise ValueError(f"frames must be between 1 and {VISION_BATCH}, got {self.frames}")
        root = Path(self.demo_root)
        missing = [name for name in (_SCRIPTS, _LIVE, "tokenizer", "om") if not (root / name).exists()]
        if missing:
            raise ValueError(f"{root} is not the InternVL deployment: missing {', '.join(missing)}")

    # -- loading ----------------------------------------------------------

    def load(self) -> None:
        """Load the tokenizer, the three OMs and the prompt. Idempotent.

        Separate from ``observe`` so a deployment can pay the ~20 s cost at
        startup and have the first round cost the same as every later one.
        """
        if self._loaded:
            return
        try:
            self._load()
        except Exception as error:  # noqa: BLE001 - reported, not raised, see module docstring
            self.last_error = f"{type(error).__name__}: {error}"
            raise VisionBackendError(self.last_error) from error
        self._loaded = True

    def _load(self) -> None:
        import numpy as np

        root = Path(self.demo_root).resolve()
        for sub in (_SCRIPTS, _LIVE):
            path = str(root / sub)
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            import aclruntime  # noqa: F401 - imported for the diagnosis below
        except ImportError as error:
            # The CANN runtime is a binary extension installed into the system
            # site-packages, and a venv does not inherit those. Observed on the
            # board: every round reported MODEL_NOT_READY with a bare
            # ModuleNotFoundError, which says nothing about what to do. Name the
            # fix here, once, where the failure happens.
            raise ImportError(
                f"{error}. The CANN runtime lives in the system site-packages, which this "
                "interpreter does not include. Source the toolkit's set_env.sh and add it, e.g. "
                "PYTHONPATH=/usr/local/lib64/python3.11/site-packages:$PYTHONPATH"
            ) from error

        import live_chat
        from llm_host import Generator
        from qwen_detok import Detokenizer
        from qwen_tok import Tokenizer
        from run_llm_om import OM, om_backend, session

        # aclruntime writes ~20 lines per inference to fd 1 and log_level does
        # not suppress it. Left unredirected it drowns the node's own logging.
        live_chat.quiet_native_output(os.devnull)
        self._preprocess = live_chat.preprocess

        tok_dir = str(root / "tokenizer")
        self._tokenizer = Tokenizer(tok_dir)
        self._detokenizer = Detokenizer(tok_dir)
        img_ctx = self._tokenizer.added["<IMG_CONTEXT>"]
        eos = self._tokenizer.added["<|im_end|>"]

        self._vision = om_backend(session(OM[_VISION_OM]), ["vit_embeds"])
        embed = np.load(root / _EMBED_TABLE, mmap_mode="r")
        self._generator = Generator(
            embed,
            om_backend(session(OM[_PREFILL_OM]), _LLM_OUTPUTS),
            om_backend(session(OM[_DECODE_OM]), _LLM_OUTPUTS),
            PREFILL_LEN,
            CONTEXT_LEN,
            img_ctx,
            eos,
        )
        self._prompt_ids = np.array(self._tokenizer.encode(self._prompt_text()))
        self.prompt_tokens = int(len(self._prompt_ids))

    def _prompt_text(self) -> str:
        """Our closed-schema question, with the image slots expanded.

        Built once: the prompt is fixed for a deployment, and the budget check
        in phase0.budget only holds for the shape it was measured against.
        """
        from visual_interaction.phase0 import prompts
        from visual_interaction.phase0.budget import OmShape, assess, expand_images

        shape = OmShape(
            prefill_len=PREFILL_LEN,
            context_len=CONTEXT_LEN,
            frames=self.frames,
            image_tokens_per_frame=IMAGE_TOKENS,
        )
        if self.prompt_mode == "chatter":
            from visual_interaction.session.chatter import chatter_prompt

            text = expand_images(chatter_prompt(self.frames), shape)
        else:
            scenario = prompts.observe_scenario("runtime", frames=self.frames)
            text = expand_images(scenario.text, shape)
        # Fail here rather than inside the generator: it raises on an overlong
        # prompt only after the vision OM has already run.
        tokens = len(self._tokenizer.encode(text))
        fit = assess(
            "runtime",
            prompt_tokens=tokens,
            max_new_tokens=self.max_new_tokens,
            shape=shape,
            with_images=True,
        )
        if not fit.fits:
            raise ValueError(
                f"prompt does not fit the OM shapes: {tokens} tokens + {self.max_new_tokens} new, "
                f"prefill {PREFILL_LEN}, cache {CONTEXT_LEN}"
            )
        return text

    # -- one round --------------------------------------------------------

    def observe(self, inputs: RoundInputs) -> str | None:
        """Run the model on this window. ``None`` means it could not answer.

        Every fault returns None rather than raising: the session turns that
        into MODEL_NOT_READY, which is a reportable outcome, while an exception
        from a port would take down the round that was already in progress.
        """
        import numpy as np

        try:
            self.load()
        except VisionBackendError:
            return None

        images = [frame for frame in inputs.window.frames if frame is not None]
        if len(images) < self.frames:
            self.last_error = f"window carried {len(images)} usable frames, need {self.frames}"
            return None

        try:
            # The newest frames, oldest first, so the prompt's Frame1..N order
            # matches the order they were seen in.
            chosen = images[-self.frames :]
            pixels = np.zeros((VISION_BATCH, 3, 448, 448), np.float32)
            pixels[: self.frames] = self._preprocess(chosen)
            embeds = self._vision({"pixel_values": pixels})[0]
            embeds = embeds.reshape(VISION_BATCH, IMAGE_TOKENS, -1)[: self.frames]
            result = self._generator.run(
                self._prompt_ids,
                embeds,
                max_new_tokens=self.max_new_tokens,
                keep_logits=False,
            )
        except Exception as error:  # noqa: BLE001 - see the docstring
            self.last_error = f"{type(error).__name__}: {error}"
            return None

        self.last_error = ""
        return self._detokenizer.decode(result["tokens"]).strip()


__all__ = [
    "CONTEXT_LEN",
    "IMAGE_TOKENS",
    "PREFILL_LEN",
    "VISION_BATCH",
    "InternVlVision",
    "VisionBackendError",
]
