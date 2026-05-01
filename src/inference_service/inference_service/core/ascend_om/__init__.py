"""Ascend OM backends for ``inference_service``."""

from inference_service.core.ascend_om.ACTWrapper import ACTWrapper
from inference_service.core.ascend_om.ACTWrapper_3403 import ACT3403Policy
from inference_service.core.ascend_om.OMmodel import OMmodel
from inference_service.core.ascend_om.policy_wrapper import (
    AscendOM3403PolicyWrapper,
    AscendOMPolicyWrapper,
    create_ascend_om_policy_wrapper,
    resolve_3403_worker_path,
    resolve_om_model_path,
)

__all__ = [
    "ACT3403Policy",
    "ACTWrapper",
    "AscendOM3403PolicyWrapper",
    "AscendOMPolicyWrapper",
    "OMmodel",
    "create_ascend_om_policy_wrapper",
    "resolve_3403_worker_path",
    "resolve_om_model_path",
]
