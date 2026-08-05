# DelegatedExecutorIdentity.msg - Identity of a delegated executor captured in
# the registry snapshot and re-stated on every delegated dispatch. Model
# fields (model_deployment_name, model_fingerprint, model_bundle_digest) must
# be all empty for non-model executors and all non-empty for model executors.
#
# See docs/lightweight_skill_package_registry_design_zh.md section 9.1 and 11.5.
# schema_version is 1 in v1.

uint32 schema_version
string name
string contract_version
string endpoint_kind
string endpoint_name
string configuration_digest
string model_deployment_name
string model_fingerprint
string model_bundle_digest
