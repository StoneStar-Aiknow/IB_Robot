import pytest

from semantic_mapping.migration_evidence import MigrationEvidence, validate_production_switch


def test_local_evidence_can_pass_without_claiming_hardware():
    evidence = MigrationEvidence(
        local_tests_passed=True,
        clean_build_passed=True,
        fixture_e2e_passed=True,
        grasp_compatibility_passed=True,
        service_rollback_verified=True,
        database_migration_verified=True,
    )

    assert evidence.local_gate_passed
    assert not evidence.production_gate_passed
    assert evidence.blockers() == ("hardware conformance", "hardware timing")


def test_production_switch_remains_fail_closed_without_hardware():
    with pytest.raises(RuntimeError, match="hardware conformance"):
        validate_production_switch(MigrationEvidence(local_tests_passed=True))
