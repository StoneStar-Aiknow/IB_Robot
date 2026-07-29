from embodied_common.skill_templates import DEFAULT_SKILL_TEMPLATES, get_skill_templates


def test_get_skill_templates_defaults_only_when_templates_are_absent():
    assert get_skill_templates(None) == DEFAULT_SKILL_TEMPLATES
    assert get_skill_templates({}) == {}
    assert (
        get_skill_templates(
            {"disabled_skill": {"disabled": True, "primitive_sequence": [{"primitive_name": "open_gripper"}]}}
        )
        == {}
    )


def test_get_skill_templates_excludes_explicitly_disabled_templates():
    templates = {
        "enabled_skill": {"primitive_sequence": [{"primitive_name": "open_gripper"}]},
        "disabled_skill": {
            "disabled": True,
            "primitive_sequence": [{"primitive_name": "close_gripper"}],
        },
    }

    resolved = get_skill_templates(templates)

    assert set(resolved) == {"enabled_skill"}
