from pathlib import Path
from xml.etree import ElementTree

PACKAGE_ROOT = Path(__file__).parents[1]
SROS_ROOT = PACKAGE_ROOT / "sros2"


def _permissions():
    root = ElementTree.parse(SROS_ROOT / "caller_policy.xml").getroot()
    result = {}
    for enclave in root.findall("./enclaves/enclave"):
        path = enclave.attrib["path"]
        services = {"request": set(), "reply": set()}
        actions = {"call": set(), "execute": set()}
        profile = enclave.find("./profiles/profile")
        assert profile is not None
        for node in profile:
            if node.tag == "services":
                for service in node.findall("service"):
                    for direction in node.attrib:
                        services[direction].add(service.text)
            if node.tag == "actions":
                for action in node.findall("action"):
                    for direction in node.attrib:
                        actions[direction].add(action.text)
        result[path] = {"services": services, "actions": actions}
    return result


def test_caller_policy_has_no_wildcard_motion_permissions():
    permissions = _permissions()
    assert "/hermes_cli" in permissions
    assert "/operator" in permissions
    assert "/embodied/execute_primitive" not in permissions["/hermes_cli"]["actions"]["call"]
    assert "/embodied/reload_skill_catalog" not in permissions["/hermes_cli"]["services"]["request"]
    assert "/embodied/reload_skill_catalog" in permissions["/operator"]["services"]["request"]
    for enclave in permissions.values():
        for values in (*enclave["services"].values(), *enclave["actions"].values()):
            assert "/*" not in values


def test_governance_requires_authenticated_participants():
    root = ElementTree.parse(SROS_ROOT / "governance.xml").getroot()
    domain = root.find("./domain_access_rules/domain_rule")
    assert domain is not None
    assert domain.findtext("allow_unauthenticated_participants") == "false"
    assert domain.findtext("enable_join_access_control") == "true"
