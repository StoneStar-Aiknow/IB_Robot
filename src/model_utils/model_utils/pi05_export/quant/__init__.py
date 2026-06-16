# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""PI05 msModelSlim W8A8 quantization toolkit.

Modules
-------
- :mod:`w8a8_common`   — shared msModelSlim patches, ONNX graph surgery,
  the quantization driver, and the Route-A int8 transplant. No model-specific
  logic lives here.
- :mod:`quantize_vlm`  — VLM (gemma_2b) calibration-data builder + CLI.
- :mod:`quantize_ae`   — Action Expert (gemma_300m) calibration-data builder + CLI.

Both entry points are thin: they only build calibration data and pick default
fp16-exclusion regexes, then delegate everything else to
:func:`w8a8_common.run_msmodelslim_w8a8`.
"""
