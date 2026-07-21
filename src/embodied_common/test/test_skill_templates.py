from embodied_common.skill_templates import get_skill_templates


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
