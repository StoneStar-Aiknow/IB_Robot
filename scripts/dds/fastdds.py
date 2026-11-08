"""Configure network discovery without replacing Fast DDS' builtin transports."""

from xml.etree import ElementTree as ET

VERSION = "2.6.11"
NAMESPACE = "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"
MAX_INITIAL_PEER_INDEX = 32
# Socket buffers for the builtin UDP transport. Left unset, Fast DDS takes the
# kernel default (often 208 KiB): a reliable sample larger than that, such as a
# ~590 KB grasp-head capture reply, loses fragments on its first send and waits
# for the 3 s heartbeat to be repaired (measured on the X2 Orin, 2026-11-07).
# The kernel still caps each request at net.core.[wr]mem_max.
SOCKET_BUFFER_BYTES = 2 * 1024 * 1024


def render(network, binding, domain_id=0):
    """Input is normalized YAML plus a verified local interface binding."""
    mode = network["discovery"]
    address = binding["address"]
    root = ET.Element("profiles", {"xmlns": NAMESPACE})
    name = {"local_only": "ibrobot_local", "lan_multicast": "ibrobot_lan", "static_peers": "ibrobot_static"}[mode]
    participant = ET.SubElement(root, "participant", {"profile_name": name, "is_default_profile": "true"})
    rtps = ET.SubElement(participant, "rtps")
    ET.SubElement(rtps, "sendSocketBufferSize").text = str(SOCKET_BUFFER_BYTES)
    ET.SubElement(rtps, "listenSocketBufferSize").text = str(SOCKET_BUFFER_BYTES)

    def locators(parent, name, addresses):
        group = ET.SubElement(parent, name)
        for ip in addresses:
            locator = ET.SubElement(group, "locator")
            ET.SubElement(ET.SubElement(locator, "udpv4"), "address").text = ip
        return group

    if mode != "lan_multicast":
        builtin = ET.SubElement(rtps, "builtin")
        locators(builtin, "metatrafficUnicastLocatorList", [address])
        addresses = [address, *network.get("peers", [])]
        initial_peers = locators(builtin, "initialPeersList", addresses)
        # The builtin UDP transport probes indices 0..3 for port-free peers.
        # Extend discovery only; leave builtin transports and data locators alone.
        for ip in addresses:
            for index in range(4, MAX_INITIAL_PEER_INDEX + 1):
                udp = ET.SubElement(ET.SubElement(initial_peers, "locator"), "udpv4")
                ET.SubElement(udp, "address").text = ip
                ET.SubElement(udp, "port").text = str(7400 + 250 * domain_id + 10 + 2 * index)
    # Leave data locators unset. Fast DDS supplies both builtin UDP and SHM
    # default unicast locators when it constructs the participant; listing only
    # a UDP locator here would suppress the local SHM data path.
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
