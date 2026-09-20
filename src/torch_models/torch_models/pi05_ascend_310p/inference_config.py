"""Model-owned deployment parameters; independent of Torch and ROS imports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


def _object(value: object, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return dict(value)


@dataclass(frozen=True)
class DenoisingConfig:
    solver: str = "euler"
    schedule: str = "uniform"
    steps: int = 10

    def __post_init__(self) -> None:
        if self.solver not in ("euler", "ab2"):
            raise ValueError("PI05 denoising.solver must be euler or ab2")
        if self.schedule != "uniform":
            raise ValueError("PI05 Ascend310P currently supports only a uniform schedule")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("PI05 denoising.steps must be a positive integer")
        if self.solver == "ab2" and self.steps < 2:
            raise ValueError("PI05 AB2 requires at least two network evaluations")

    @property
    def timesteps(self) -> tuple[float, ...]:
        return tuple(1.0 - step / self.steps for step in range(self.steps)) + (0.0,)

    @classmethod
    def parse(cls, value: object) -> DenoisingConfig:
        return cls(**_object(value, {"solver", "schedule", "steps"}, "PI05 denoising"))


@dataclass(frozen=True)
class PI05AscendInferenceConfig:
    denoising: DenoisingConfig = field(default_factory=DenoisingConfig)
    language_capacity: int | None = None
    vision_ffn: bool = False
    image_preprocessing: str = "native"
    action_output_device: str = "npu"

    def __post_init__(self) -> None:
        if not isinstance(self.denoising, DenoisingConfig):
            raise ValueError("PI05 denoising must be a DenoisingConfig")
        if self.language_capacity is not None and (
            type(self.language_capacity) is not int or self.language_capacity < 1
        ):
            raise ValueError("PI05 language_capacity must be null or a positive integer")
        if type(self.vision_ffn) is not bool:
            raise ValueError("PI05 vision_ffn must be a boolean")
        if self.image_preprocessing not in ("native", "layout_aware"):
            raise ValueError("PI05 image_preprocessing must be native or layout_aware")
        if self.action_output_device not in ("npu", "cpu"):
            raise ValueError("PI05 action_output_device must be npu or cpu")

    @classmethod
    def parse(cls, value: object) -> PI05AscendInferenceConfig:
        values = _object(
            value,
            {"denoising", "language_capacity", "vision_ffn", "image_preprocessing", "action_output_device"},
            "PI05 extra_parameters",
        )
        if "denoising" in values:
            values["denoising"] = DenoisingConfig.parse(values["denoising"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def cache_key(self) -> str:
        # Derived entirely from effective settings. No user checksum or file
        # verification is required for inline deployment parameters.
        return (
            f"{self.denoising.solver}-{self.denoising.schedule}-{self.denoising.steps}"
            f"-capacity-{self.language_capacity}-vision-ffn-{int(self.vision_ffn)}"
            f"-images-{self.image_preprocessing}-output-{self.action_output_device}"
        )


def deployment_parameters(mode: str) -> dict[str, Any]:
    """Return an independent settings snapshot for the model's named presets."""
    if mode not in ("fidelity", "fast"):
        raise ValueError(f"unknown PI05 Ascend310P deployment mode: {mode!r}")
    return PI05AscendInferenceConfig(
        denoising=DenoisingConfig(
            solver="euler" if mode == "fidelity" else "ab2", steps=10 if mode == "fidelity" else 3
        ),
        language_capacity=48 if mode == "fast" else 200,
        vision_ffn=True,
        image_preprocessing="layout_aware",
        action_output_device="cpu",
    ).to_dict()
