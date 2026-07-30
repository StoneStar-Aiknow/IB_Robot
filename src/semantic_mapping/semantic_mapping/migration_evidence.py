"""Evidence checklist for semantic-mapping backend migration and rollback."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MigrationEvidence:
    """Operator-recorded evidence required before changing the production default."""

    local_tests_passed: bool = False
    clean_build_passed: bool = False
    fixture_e2e_passed: bool = False
    grasp_compatibility_passed: bool = False
    service_rollback_verified: bool = False
    database_migration_verified: bool = False
    hardware_conformance_passed: bool = False
    hardware_timing_passed: bool = False

    @property
    def local_gate_passed(self) -> bool:
        return all(
            (
                self.local_tests_passed,
                self.clean_build_passed,
                self.fixture_e2e_passed,
                self.grasp_compatibility_passed,
                self.service_rollback_verified,
                self.database_migration_verified,
            )
        )

    @property
    def production_gate_passed(self) -> bool:
        return self.local_gate_passed and self.hardware_conformance_passed and self.hardware_timing_passed

    def blockers(self) -> tuple[str, ...]:
        checks = (
            ("local tests", self.local_tests_passed),
            ("clean build", self.clean_build_passed),
            ("fixture end-to-end", self.fixture_e2e_passed),
            ("grasp compatibility", self.grasp_compatibility_passed),
            ("service rollback", self.service_rollback_verified),
            ("database migration", self.database_migration_verified),
            ("hardware conformance", self.hardware_conformance_passed),
            ("hardware timing", self.hardware_timing_passed),
        )
        return tuple(name for name, passed in checks if not passed)


def validate_production_switch(evidence: MigrationEvidence) -> None:
    """Reject a production-default switch until every gate has evidence."""
    blockers = evidence.blockers()
    if blockers:
        raise RuntimeError("production backend switch is blocked by: " + ", ".join(blockers))
