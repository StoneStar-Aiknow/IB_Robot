"""Exceptions raised while loading or validating inference manifests."""


class ManifestError(ValueError):
    """Base error for manifest and policy-bundle validation failures."""


class ManifestPathError(ManifestError):
    """A manifest path is unsafe or cannot be resolved inside the bundle."""


class ManifestIntegrityError(ManifestError):
    """A declared file or canonical digest does not match bundle contents."""


class ManifestValidationError(ManifestError):
    """Manifest structure or semantic validation failed."""
