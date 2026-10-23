"""Configure network discovery without replacing CycloneDDS transport defaults."""

from xml.etree import ElementTree as ET

VERSION = "0.10.5"
NAMESPACE = "https://cdds.io/config"
MAX_AUTO_PARTICIPANT_INDEX = 32


def render(network, binding):
    """Input is normalized YAML plus a verified local interface binding."""
    mode = network["discovery"]
    attributes = {"address": binding["address"]}
    if mode != "local_only":
        attributes["name"] = binding["interface"]
    root = ET.Element("CycloneDDS", {"xmlns": NAMESPACE})
    domain = ET.SubElement(root, "Domain", {"Id": "any"})
    general = ET.SubElement(domain, "General")
    ET.SubElement(ET.SubElement(general, "Interfaces"), "NetworkInterface", attributes)
    if mode != "lan_multicast":
        ET.SubElement(general, "AllowMulticast").text = "false"
    if mode != "lan_multicast":
        discovery = ET.SubElement(domain, "Discovery")
        ET.SubElement(discovery, "MaxAutoParticipantIndex").text = str(MAX_AUTO_PARTICIPANT_INDEX)
        # CycloneDDS 0.10.5 defaults ParticipantIndex to "none"; without
        # multicast, unicast peers need the indexed discovery ports instead.
        ET.SubElement(discovery, "ParticipantIndex").text = "auto"
        group = ET.SubElement(discovery, "Peers")
        for address in (binding["address"], *network.get("peers", [])):
            ET.SubElement(group, "Peer", {"Address": address})
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
