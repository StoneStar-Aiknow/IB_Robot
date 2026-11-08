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

look_say asks up to three short questions about one picture, so it does not go
through ``observe``. It uses the split form instead: ``prepare`` crops the
newest frame and runs the vision OM once; ``ask`` runs prefill and decode for
one registered question against those image features. The features are a plain
host array (the OM backend copies its output off the device), and the
generator only reads them when it builds a prompt, so any number of questions,
in any order, can reuse one ``Seen``. The questions are fixed per deployment,
registered before ``load`` and tokenized and budget-checked there, once; a
question that does not fit the OM shapes is remembered and refused by ``ask``.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping, Sequence
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


class QuestionError(ValueError):
    """``ask`` named a question that was never registered or failed its budget check."""


@dataclass(frozen=True)
class Seen:
    """One picture, run through the vision OM once.

    ``embeds`` is the read-only host array of image features, shape
    ``(1, IMAGE_TOKENS, hidden)``; ``image`` is the 448² BGR input it was
    computed from; ``crop_fallback`` is true when a crop was asked for and the
    whole frame was used instead; ``seconds`` covers crop, preprocessing and
    the vision OM.
    """

    embeds: object
    image: object
    crop_fallback: bool
    seconds: float


@dataclass(frozen=True)
class Answer:
    """One question's reply. ``tokens`` counts generated tokens, end mark included."""

    text: str
    seconds: float
    tokens: int


def question_prompt(text: str) -> str:
    """The prompt for one look_say question, image slot unexpanded.

    A plain question is wrapped the way chatter wraps its own (its system
    line, one image slot, the question), since that form is what the board
    has run. A text that is already a chat prompt is taken as it is and must
    carry exactly one ``<image>``.
    """
    from visual_interaction.phase0.prompts import chat, frame_header

    if text.startswith("<|im_start|>"):
        return text
    from visual_interaction.session.chatter import CHATTER_SYSTEM

    return chat(frame_header(1) + text, system=CHATTER_SYSTEM)


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
    _rt_infer_warmups: int = field(default=0, init=False)
    # look_say: key -> (text, max_new_tokens) as registered; key -> token ids
    # for the ones that fit; key -> why not for the ones that did not.
    _questions: dict = field(default_factory=dict, init=False)
    _question_ids: dict = field(default_factory=dict, init=False)
    question_failures: dict = field(default_factory=dict, init=False)
    # One batch buffer for prepare: only row 0 is ever written, the padding
    # rows stay zero, so a call allocates no new batch.
    _pixels: object = field(default=None, init=False)
    _mean: object = field(default=None, init=False)
    _std: object = field(default=None, init=False)

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
        # The demo's normalisation constants, so prepare's preprocessing is
        # live_chat.preprocess minus the JPEG decode and the 448² resize that
        # look_crop already did with the same PIL bicubic.
        self._mean = live_chat.MEAN
        self._std = live_chat.STD
        self._pixels = np.zeros((VISION_BATCH, 3, 448, 448), np.float32)
        self._check_questions(dict(self._questions))

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

    # -- look_say questions -----------------------------------------------

    def register_questions(self, questions: Mapping[str, Sequence[object]]) -> None:
        """Register ``{key: (text, max_new_tokens)}`` for :meth:`ask`.

        Meant to be called before :meth:`load`, which tokenizes and
        budget-checks every question once. Registering after a load checks the
        new ones at once. A malformed entry raises here; one that does not fit
        the OM shapes is recorded in :attr:`question_failures` and refused by
        :meth:`ask`.
        """
        checked = {}
        for key, entry in questions.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"question key must be a non-empty string, got {key!r}")
            try:
                text, max_new_tokens = entry
            except (TypeError, ValueError) as error:
                raise ValueError(f"question {key!r} must be (text, max_new_tokens), got {entry!r}") from error
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"question {key!r} has no text")
            if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
                raise ValueError(f"question {key!r}: max_new_tokens must be a positive int, got {max_new_tokens!r}")
            checked[key] = (text, max_new_tokens)
        self._questions.update(checked)
        if self._loaded:
            self._check_questions(checked)

    def _check_questions(self, questions: Mapping[str, tuple[str, int]]) -> None:
        """Tokenize each question with one image slot and hold it to the OM shapes."""
        import numpy as np

        from visual_interaction.phase0.budget import OmShape, assess, check_image_slots, expand_images

        shape = OmShape(prefill_len=PREFILL_LEN, context_len=CONTEXT_LEN, frames=1, image_tokens_per_frame=IMAGE_TOKENS)
        for key, (text, max_new_tokens) in questions.items():
            self._question_ids.pop(key, None)
            self.question_failures.pop(key, None)
            try:
                prompt = question_prompt(text)
                check_image_slots(key, prompt, with_images=True, shape=shape)
                ids = np.array(self._tokenizer.encode(expand_images(prompt, shape)))
                fit = assess(
                    f"question:{key}",
                    prompt_tokens=int(len(ids)),
                    max_new_tokens=max_new_tokens,
                    shape=shape,
                    with_images=True,
                )
            except Exception as error:  # noqa: BLE001 - recorded per question, see register_questions
                self.question_failures[key] = f"{type(error).__name__}: {error}"
                continue
            if not fit.fits:
                self.question_failures[key] = (
                    f"prompt does not fit the OM shapes: {fit.prompt_tokens} tokens + {max_new_tokens} new, "
                    f"prefill {PREFILL_LEN}, cache {CONTEXT_LEN}"
                )
                continue
            self._question_ids[key] = ids

    def prepare(
        self,
        frame_bgr: object,
        bbox: Sequence[float] | None,
        crop: bool = True,
        *,
        source_size: tuple[int, int] | None = None,
    ) -> Seen:
        """Run the vision OM once on the newest frame, cropped to the actor.

        ``frame_bgr`` is a decoded H×W×3 uint8 frame; ``bbox`` the actor box
        in its pixels (or in ``source_size`` pixels, see look_crop), or None.
        ``crop=False`` uses the whole frame. Raises :class:`VisionBackendError`
        when the model cannot be loaded or the vision OM fails, and ValueError
        for a frame that is not an image.
        """
        import numpy as np

        from visual_interaction.session import look_crop

        self.load()
        started = time.perf_counter()
        if crop:
            cut = look_crop.crop_upper_body(frame_bgr, bbox, source_size=source_size)
            image, fallback = cut.image, cut.fallback
        else:
            image, fallback = look_crop.whole_frame(frame_bgr), False
        try:
            rgb = image[:, :, ::-1]
            # Same operations, in the same order, as live_chat.preprocess.
            self._pixels[0] = (np.asarray(rgb, np.float32).transpose(2, 0, 1) / 255.0 - self._mean) / self._std
            out = self._vision({"pixel_values": self._pixels})[0]
            embeds = np.array(out.reshape(VISION_BATCH, IMAGE_TOKENS, -1)[:1])
        except Exception as error:  # noqa: BLE001 - one failure type for the caller
            self.last_error = f"{type(error).__name__}: {error}"
            raise VisionBackendError(self.last_error) from error
        # Shared by every ask on this picture: nobody may write into it.
        embeds.flags.writeable = False
        self.last_error = ""
        # The same readiness signal observe gives an rt_infer launch, counted
        # per picture: a look_say node never calls observe.
        self._rt_infer_warmups += 1
        if self._rt_infer_warmups == 3 and os.environ.get("RTINFER_LAUNCH_TOKEN"):
            from inference_service.rt_infer_client import mark_runtime_ready

            mark_runtime_ready()
        return Seen(embeds=embeds, image=image, crop_fallback=fallback, seconds=time.perf_counter() - started)

    def ask(self, seen: Seen, key: str) -> Answer:
        """Ask one registered question about ``seen``. Prefill and decode only.

        Stateless between calls: any number of keys, repeated or not, may be
        asked of one ``Seen``. Raises :class:`QuestionError` for a key that is
        unknown or failed its budget check, :class:`VisionBackendError` when
        generation fails.
        """
        if key in self.question_failures:
            raise QuestionError(f"question {key!r} was refused at load: {self.question_failures[key]}")
        ids = self._question_ids.get(key)
        if ids is None:
            if key in self._questions:
                raise QuestionError(f"question {key!r} is not loaded yet: call load() or prepare() first")
            raise QuestionError(f"question {key!r} is not registered")
        max_new_tokens = self._questions[key][1]
        started = time.perf_counter()
        try:
            result = self._generator.run(ids, seen.embeds, max_new_tokens=max_new_tokens, keep_logits=False)
            text = self._detokenizer.decode(result["tokens"]).strip()
        except Exception as error:  # noqa: BLE001 - one failure type for the caller
            self.last_error = f"{type(error).__name__}: {error}"
            raise VisionBackendError(self.last_error) from error
        return Answer(text=text, seconds=time.perf_counter() - started, tokens=len(result["tokens"]))

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
        self._rt_infer_warmups += 1
        import os

        if self._rt_infer_warmups == 3 and os.environ.get("RTINFER_LAUNCH_TOKEN"):
            from inference_service.rt_infer_client import mark_runtime_ready

            mark_runtime_ready()
        return self._detokenizer.decode(result["tokens"]).strip()


__all__ = [
    "CONTEXT_LEN",
    "IMAGE_TOKENS",
    "PREFILL_LEN",
    "VISION_BATCH",
    "Answer",
    "InternVlVision",
    "QuestionError",
    "Seen",
    "VisionBackendError",
    "question_prompt",
]
