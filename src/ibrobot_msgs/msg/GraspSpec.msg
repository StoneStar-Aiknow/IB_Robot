# Optional grasp intent carried by PickObject (schema_version 1).
# schema_version=0 means no spec: the executor chooses as before. Every field is
# optional and empty means "executor's choice". An executor that cannot honour
# a non-empty field (see its capability declaration) rejects the goal with
# GRASP_SPEC_UNSUPPORTED; it never ignores a field silently.
# An unknown schema_version is GRASP_SPEC_UNSUPPORTED too; a malformed spec is
# GRASP_SPEC_INVALID. Validation: embodied_common.pick_contract.check_grasp_spec.
uint32 schema_version
# The object part to grasp, as an open-vocabulary phrase ("handle").
string part
# Parts the hand must not touch ("rim", "spout").
string[] avoid_parts
# A grasp type the effector declares ("pinch", "power").
string grasp_type
# Fingers that must make contact, by the effector's declared finger names.
string[] fingers
# Target grip force in newtons; 0 keeps the effector's default.
float64 target_force_n
