"""Pure catalog generation management without execution or lease state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from skill_catalog.models import SkillRegistryError, SkillRuntimeBundle, SkillSnapshot


@dataclass(frozen=True)
class RegistryActivation:
    bundle: SkillRuntimeBundle
    changed: bool
    changed_skills: tuple[str, ...]


class SkillRegistry:
    def __init__(self, *, registry_epoch: str | None = None, max_unretained_history: int = 2) -> None:
        self.registry_epoch = registry_epoch or str(uuid.uuid4())
        self.max_unretained_history = max_unretained_history
        self._current: SkillRuntimeBundle | None = None
        self._bundles: dict[int, SkillRuntimeBundle] = {}
        self._retention_counts: dict[int, int] = {}

    @property
    def current(self) -> SkillRuntimeBundle:
        if self._current is None:
            raise SkillRegistryError("skill registry has not been activated")
        return self._current

    def activate(self, snapshot: SkillSnapshot) -> RegistryActivation:
        previous = self._current
        if previous is not None and _same_snapshot(previous.snapshot, snapshot):
            return RegistryActivation(previous, False, ())
        generation = 1 if previous is None else previous.generation + 1
        bundle = SkillRuntimeBundle.from_snapshot(
            snapshot,
            registry_epoch=self.registry_epoch,
            generation=generation,
        )
        self._bundles[generation] = bundle
        self._retention_counts.setdefault(generation, 0)
        self._current = bundle
        changed_skills = _changed_skills(previous.snapshot if previous else None, snapshot)
        self._collect_history()
        return RegistryActivation(bundle, True, changed_skills)

    def get(self, *, registry_epoch: str = "", generation: int = 0) -> SkillRuntimeBundle:
        current = self.current
        if registry_epoch and registry_epoch != self.registry_epoch:
            raise SkillRegistryError("registry epoch does not match", code="SKILL_REGISTRY_EPOCH_MISMATCH")
        if generation == 0:
            return current
        try:
            return self._bundles[generation]
        except KeyError as exc:
            raise SkillRegistryError("snapshot generation is not retained", code="SKILL_SNAPSHOT_NOT_RETAINED") from exc

    def retain(self, generation: int) -> SkillRuntimeBundle:
        bundle = self.get(registry_epoch=self.registry_epoch, generation=generation)
        self._retention_counts[generation] = self._retention_counts.get(generation, 0) + 1
        return bundle

    def release(self, generation: int) -> None:
        count = self._retention_counts.get(generation, 0)
        if count <= 0:
            return
        self._retention_counts[generation] = count - 1
        self._collect_history()

    @property
    def retained_generations(self) -> tuple[int, ...]:
        return tuple(sorted(self._bundles))

    def _collect_history(self) -> None:
        if self._current is None:
            return
        unretained = [
            generation
            for generation in sorted(self._bundles)
            if generation != self._current.generation and self._retention_counts.get(generation, 0) == 0
        ]
        for generation in unretained[: -self.max_unretained_history or None]:
            self._bundles.pop(generation, None)
            self._retention_counts.pop(generation, None)


def _same_snapshot(left: SkillSnapshot, right: SkillSnapshot) -> bool:
    return (
        left.registry_digest == right.registry_digest
        and left.capability_digest == right.capability_digest
        and left.provenance_digest == right.provenance_digest
    )


def _changed_skills(previous: SkillSnapshot | None, current: SkillSnapshot) -> tuple[str, ...]:
    if previous is None:
        return current.enabled_skill_names
    names = set(previous.enabled_skill_names) | set(current.enabled_skill_names)
    changed = []
    for name in names:
        if (
            previous.templates.get(name) != current.templates.get(name)
            or previous.capability_view.get(name) != current.capability_view.get(name)
            or previous.provenance.get("skill_package_digests", {}).get(name)
            != current.provenance.get("skill_package_digests", {}).get(name)
        ):
            changed.append(name)
    return tuple(sorted(changed))
