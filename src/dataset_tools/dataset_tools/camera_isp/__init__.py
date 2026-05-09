"""Camera ISP color-calibration core algorithms.

Pure-numpy module — no ROS / cv2 dependencies. Used by
:mod:`dataset_tools.camera_isp_calibrator` (GUI/ROS layer) and by unit tests.

Public surface:
    solver.auto_match_lab(...)
    solver.manual_match_neutral(...)
    solver.manual_match_ref(...)
    solver.manual_match_patches(...)
    color_space.bgr_to_lab(...)
    color_space.bgr_to_xyz_chromaticity(...)
    color_space.mccamy_kelvin(...)
    lut.lookup_kelvin(rg, bg)
    color_search.search_KCS(...)            # unified K/C/Sat search driver
    color_search.kmeans_signature_lab(...)  # Lab cluster signature
    color_search.nn_match_signatures(...)   # Hungarian ΔE2000 match
    color_search.cost_24card / cost_ref_cluster / cost_manual_roi
    color_search.cost_palette_swd            # AUTO v2: chroma-weighted SWD on (a*,b*)
    color_search.sliced_wasserstein_2d       # building block for AUTO v2
"""
