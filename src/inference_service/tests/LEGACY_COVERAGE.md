# Legacy Inference Test Coverage

This map records where behavior from the removed wrapper/coordinator test suites
is verified in the unified manifest, pipeline, codec, and backend architecture.

| Removed test surface | Unified replacement coverage |
|---|---|
| `test_inference_coordinator.py` preprocessing, inference, postprocessing, chunk metadata, timing, routing, reset, and shutdown | `test_inference_pipeline.py`, `test_pipeline_processors.py`, `test_pure_inference_engine.py`, `test_cpu_smoke_bundle.py` |
| `test_compiled_policy.py` policy-family input mapping, dtype/layout conversion, execution ordering, action selection, artifact resolution, and lazy SDK loading | `test_policy_codecs.py`, `test_codec_execution_plan.py`, `test_inference_manifest.py`, `test_backend_contract.py`, `test_ascend_backend.py`, `test_rknn_backend.py`, `test_hmm_backend.py` |
| `test_ascend_om.py` ACT and PI0.5 OM execution and removed backend-name rejection | `test_ascend_backend.py`, `test_inference_pipeline.py`, `test_backend_contract.py`, `test_inference_manifest.py` |
| `test_actwrapper_3403.py` worker framing, output routing, restart, and shutdown | `test_hisilicon_protocol.py`, `test_hisilicon_backend.py` |
| `test_hmm_pi05.py` and `test_hmm_policy_wrapper.py` multi-module HMM execution and supported policy selection | `test_hmm_backend.py`, `test_codec_execution_plan.py`, `test_backend_contract.py` |
| `test_rknn_policy_wrapper.py` RKNN artifact loading, image layout, action output, and cleanup | `test_rknn_backend.py`, `test_policy_codecs.py`, `test_backend_contract.py` |
| `test_policy_config.py` read-only LeRobot config loading and in-memory device placement | `test_bundle_metadata.py`, `test_pipeline_processors.py`, `test_torch_backend.py` |

Positive coverage for removed aliases such as `ascend_om`, `ascend_om_3403`, and
backend-valued `device` settings is intentionally not preserved. Negative alias
rejection remains in the manifest and backend contract suites.
