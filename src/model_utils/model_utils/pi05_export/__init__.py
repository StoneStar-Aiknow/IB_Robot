"""PI05 ONNX-export, monkey-patch and dump tooling for Ascend OM deployment.

This subpackage hosts the offline tools used to split the lerobot PI05 policy into
its VLM and Action Expert parts, export them to ONNX (for Ascend OM conversion),
apply the export/perf monkey patches, and dump intermediate tensors for debugging.

Runtime OM inference lives in ``inference_service.core.ascend_om.pi05``.
"""
