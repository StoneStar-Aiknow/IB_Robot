import pytest

from semantic_mapping.workflow_readiness import (
    manipulation_confirmation,
    navigation_staging,
    offline_map_construction,
    online_map_construction,
    read_only_diagnostics,
    structured_query,
    text_query,
)

OFFLINE_READY = {
    "sam2_ready": True,
    "sam2_identity_compatible": True,
    "ram_plus_ready": True,
    "ram_plus_identity_compatible": True,
    "siglip2_image_ready": True,
    "siglip2_image_identity_compatible": True,
    "rgbd_input_ready": True,
    "timestamped_tf_ready": True,
    "localization_ready": True,
}
ONLINE_READY = {
    **OFFLINE_READY,
    "active_map_identity_compatible": True,
    "authoritative_map_odom": True,
    "cloud_map_ready": True,
    "queue_write_allowed": True,
}
NAVIGATION_READY = {
    "object_action_admissible": True,
    "active_map_identity_compatible": True,
    "localization_ready": True,
    "authoritative_map_odom": True,
    "timestamped_tf_ready": True,
    "footprint_ready": True,
    "obstacle_map_ready": True,
    "reachability_ready": True,
}


@pytest.mark.parametrize("failed_gate", tuple(OFFLINE_READY))
def test_offline_map_construction_matrix_reports_each_failed_gate(failed_gate):
    values = {**OFFLINE_READY, failed_gate: False}
    readiness = offline_map_construction(**values)

    assert not readiness.ready
    assert [item.gate for item in readiness.evidence if not item.satisfied] == [failed_gate]
    assert readiness.reason


@pytest.mark.parametrize("failed_gate", tuple(ONLINE_READY))
def test_online_map_construction_matrix_includes_input_model_slam_and_write_gates(failed_gate):
    values = {**ONLINE_READY, failed_gate: False}
    readiness = online_map_construction(**values)

    assert not readiness.ready
    assert failed_gate in {item.gate for item in readiness.evidence if not item.satisfied}


def test_map_construction_reports_all_failures_instead_of_short_circuiting():
    readiness = offline_map_construction(**{name: False for name in OFFLINE_READY})

    assert len(readiness.reasons) == len(OFFLINE_READY)
    assert len(readiness.evidence) == len(OFFLINE_READY)


@pytest.mark.parametrize("failed_gate", ("database_readable", "database_compatible"))
def test_structured_query_requires_only_a_readable_compatible_database(failed_gate):
    values = {"database_readable": True, "database_compatible": True, failed_gate: False}
    readiness = structured_query(**values)

    assert not readiness.ready
    assert len(readiness.reasons) == 1


@pytest.mark.parametrize(
    "failed_gate",
    ("database_readable", "database_compatible", "siglip2_text_ready", "embedding_space_compatible"),
)
def test_text_query_matrix_adds_text_runtime_and_embedding_space(failed_gate):
    values = {
        "database_readable": True,
        "database_compatible": True,
        "siglip2_text_ready": True,
        "embedding_space_compatible": True,
        failed_gate: False,
    }
    readiness = text_query(**values)

    assert not readiness.ready
    assert failed_gate in {item.gate for item in readiness.evidence if not item.satisfied}


@pytest.mark.parametrize("failed_gate", tuple(NAVIGATION_READY))
def test_navigation_staging_matrix_reports_action_and_navigation_contract_gates(failed_gate):
    values = {**NAVIGATION_READY, failed_gate: False}
    readiness = navigation_staging(**values)

    assert not readiness.ready
    assert failed_gate in {item.gate for item in readiness.evidence if not item.satisfied}


@pytest.mark.parametrize(
    "failed_gate",
    ("object_confirmation_admissible", "gdino_ready", "confirmation_sam2_ready", "confirmation_result_fresh"),
)
def test_manipulation_confirmation_adds_object_and_both_model_gates(failed_gate):
    values = {
        "object_confirmation_admissible": True,
        "gdino_ready": True,
        "confirmation_sam2_ready": True,
        "confirmation_result_fresh": True,
        failed_gate: False,
    }
    readiness = manipulation_confirmation(navigation=navigation_staging(**NAVIGATION_READY), **values)

    assert not readiness.ready
    assert failed_gate in {item.gate for item in readiness.evidence if not item.satisfied}


def test_manipulation_inherits_navigation_failure_and_rejects_unrelated_evidence():
    navigation = navigation_staging(**{**NAVIGATION_READY, "localization_ready": False})
    readiness = manipulation_confirmation(
        navigation=navigation,
        object_confirmation_admissible=True,
        gdino_ready=True,
        confirmation_sam2_ready=True,
        confirmation_result_fresh=True,
    )

    assert not readiness.ready
    assert "global localization" in readiness.reason
    with pytest.raises(ValueError, match="navigation staging"):
        manipulation_confirmation(
            navigation=structured_query(database_readable=True, database_compatible=True),
            object_confirmation_admissible=True,
            gdino_ready=True,
            confirmation_sam2_ready=True,
            confirmation_result_fresh=True,
        )


def test_manipulation_fails_closed_when_confirmation_has_not_run():
    readiness = manipulation_confirmation(
        navigation=navigation_staging(**NAVIGATION_READY),
        object_confirmation_admissible=True,
        gdino_ready=False,
        confirmation_sam2_ready=False,
        confirmation_result_fresh=False,
    )

    assert not readiness.ready
    assert "fresh manipulation confirmation has not run" in readiness.reason


def test_structured_and_diagnostic_workflows_do_not_depend_on_inference():
    assert structured_query(database_readable=True, database_compatible=True).ready
    diagnostic = read_only_diagnostics(database_diagnostic_open=True)

    assert diagnostic.ready
    assert [item.gate for item in diagnostic.evidence] == ["database_diagnostic_open"]


def test_all_workflows_are_ready_when_every_applicable_gate_passes():
    offline = offline_map_construction(**OFFLINE_READY)
    online = online_map_construction(**ONLINE_READY)
    structured = structured_query(database_readable=True, database_compatible=True)
    text = text_query(
        database_readable=True,
        database_compatible=True,
        siglip2_text_ready=True,
        embedding_space_compatible=True,
    )
    navigation = navigation_staging(**NAVIGATION_READY)
    manipulation = manipulation_confirmation(
        navigation=navigation,
        object_confirmation_admissible=True,
        gdino_ready=True,
        confirmation_sam2_ready=True,
        confirmation_result_fresh=True,
    )
    diagnostic = read_only_diagnostics(database_diagnostic_open=True)

    assert all(item.ready for item in (offline, online, structured, text, navigation, manipulation, diagnostic))
