"""Prompt token budget against the fixed shapes of the board OMs.

OMs take no dynamic shapes, so the demo pads every prompt to a fixed prefill
length ``P`` and keeps a fixed KV cache of ``L`` positions. Its generator
refuses a prompt of ``n`` tokens unless ``n <= P`` and
``n + max_new_tokens - 1 <= L``; this module applies the same two bounds before
a prompt ever reaches the board, so an overflow fails here and not mid-round.

Two consequences the numbers below make visible:

* Prefill costs the same for any ``n <= P`` — it always runs at ``P``. A
  shorter prompt saves nothing in prefill; only the decode budget grows.
* With images, ``frames × image_tokens_per_frame`` positions are spent before
  any text. Four frames of 256 leave ``P - 1024`` for the chat template, the
  rules, the A/B/C memory and the candidates together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

IMAGE_PLACEHOLDER = "<image>"
IMAGE_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class BudgetError(ValueError):
    """A prompt does not fit the OM shapes, or its image slots are wrong."""


@dataclass(frozen=True)
class OmShape:
    """The fixed shapes one VL export was compiled with.

    Defaults are the board export the demo runs (4 × 448² frames, 256 image
    tokens each, P=1280, L=1344). They describe that export only: a new export
    must state its own, and ``vl_segments`` records the shapes it ran with.
    """

    prefill_len: int = 1280
    context_len: int = 1344
    frames: int = 4
    image_tokens_per_frame: int = 256

    def __post_init__(self) -> None:
        if min(self.prefill_len, self.context_len, self.frames, self.image_tokens_per_frame) <= 0:
            raise ValueError("every shape must be positive")
        if self.context_len < self.prefill_len:
            raise ValueError("the KV cache cannot be shorter than the prefill")
        if self.image_positions >= self.prefill_len:
            raise ValueError("the images alone fill the prefill")

    @property
    def image_positions(self) -> int:
        return self.frames * self.image_tokens_per_frame

    @property
    def text_capacity_with_images(self) -> int:
        return self.prefill_len - self.image_positions


def expand_images(text: str, shape: OmShape) -> str:
    """Replace each ``<image>`` with the InternVL context block of one frame.

    The block is ``<img>`` + ``<IMG_CONTEXT>`` × tokens-per-frame + ``</img>``,
    the slots the vision features are written into.
    """
    block = "<img>" + IMAGE_CONTEXT_TOKEN * shape.image_tokens_per_frame + "</img>"
    return text.replace(IMAGE_PLACEHOLDER, block)


def image_slots(text: str) -> int:
    return text.count(IMAGE_PLACEHOLDER)


@dataclass(frozen=True)
class Assessment:
    name: str
    with_images: bool
    prompt_tokens: int
    max_new_tokens: int
    shape: OmShape
    parts: tuple[tuple[str, int], ...] = ()

    @property
    def text_tokens(self) -> int:
        """Tokens left once the image slots are taken out."""
        return self.prompt_tokens - (self.shape.image_positions if self.with_images else 0)

    @property
    def fits_prefill(self) -> bool:
        return self.prompt_tokens <= self.shape.prefill_len

    @property
    def fits_cache(self) -> bool:
        # The generator's own bound: the last new token never needs a slot.
        return self.prompt_tokens + self.max_new_tokens - 1 <= self.shape.context_len

    @property
    def fits(self) -> bool:
        return self.fits_prefill and self.fits_cache

    @property
    def max_new_tokens_allowed(self) -> int:
        return max(0, self.shape.context_len - self.prompt_tokens + 1)

    def as_dict(self) -> dict[str, object]:
        sum_of_parts = sum(count for _, count in self.parts)
        return {
            "name": self.name,
            "with_images": self.with_images,
            "prompt_tokens": self.prompt_tokens,
            "text_tokens": self.text_tokens,
            "max_new_tokens": self.max_new_tokens,
            "prefill_headroom": self.shape.prefill_len - self.prompt_tokens,
            "max_new_tokens_allowed": self.max_new_tokens_allowed,
            "fits_prefill": self.fits_prefill,
            "fits_cache": self.fits_cache,
            "fits": self.fits,
            "parts": dict(self.parts),
            # Encoding parts apart and together differs where a merge spans a
            # boundary. The joint count is the one the OM sees; the per-part
            # counts say where the budget goes.
            "sum_of_parts": sum_of_parts,
            "boundary_difference": self.prompt_tokens - sum_of_parts if self.parts else 0,
        }


def assess(
    name: str,
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    shape: OmShape,
    with_images: bool,
    parts: Sequence[tuple[str, int]] | Mapping[str, int] = (),
) -> Assessment:
    if prompt_tokens <= 0:
        raise ValueError("a prompt has at least one token")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    items = tuple(parts.items()) if isinstance(parts, Mapping) else tuple(parts)
    return Assessment(
        name=name,
        with_images=with_images,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        shape=shape,
        parts=items,
    )


def check_image_slots(name: str, text: str, *, with_images: bool, shape: OmShape) -> None:
    """A prompt with images must name exactly one slot per frame the OM takes.

    The generator only finds the mismatch after the vision OM has run; finding
    it from the text costs nothing.
    """
    slots = image_slots(text)
    expected = shape.frames if with_images else 0
    if slots != expected:
        raise BudgetError(f"{name}: {slots} {IMAGE_PLACEHOLDER} slot(s), the shape needs {expected}")


def require_fit(assessments: Sequence[Assessment]) -> None:
    """Fail fast, naming every scenario and the bound it breaks."""
    broken = []
    for item in assessments:
        if not item.fits_prefill:
            broken.append(f"{item.name}: {item.prompt_tokens} tokens > prefill {item.shape.prefill_len}")
        if not item.fits_cache:
            broken.append(
                f"{item.name}: {item.prompt_tokens}+{item.max_new_tokens}-1 > KV cache {item.shape.context_len}"
            )
    if broken:
        raise BudgetError("; ".join(broken))


__all__ = [
    "IMAGE_CONTEXT_TOKEN",
    "IMAGE_PLACEHOLDER",
    "Assessment",
    "BudgetError",
    "OmShape",
    "assess",
    "check_image_slots",
    "expand_images",
    "image_slots",
    "require_fit",
]
