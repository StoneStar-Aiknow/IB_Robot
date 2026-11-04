"""A Warp module reload must not unload code that a captured CUDA graph launches.

cuRobo defines Warp kernel variants lazily in shared modules; a new variant
reloads the module. Without planner.retain_replaced_warp_modules the replay
below segfaults (Warp 1.11, 2026-10-09). Runs in a subprocess: a fault would
kill the test runner.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("warp")

SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    import torch
    import warp as wp
    from aimdk_grasp.planner import retain_replaced_warp_modules

    wp.init()
    retain_replaced_warp_modules()
    retain_replaced_warp_modules()  # idempotent

    def make_kernel(scale):  # a new kernel variant per constant, as in cuRobo
        @wp.kernel
        def scale_add(x: wp.array(dtype=wp.float32)):
            i = wp.tid()
            x[i] = x[i] + wp.float32(scale)
        return scale_add

    x = torch.zeros(1024, device="cuda")
    def launch(kernel):
        wp.launch(kernel, dim=1024, inputs=[wp.from_torch(x)], device="cuda:0",
                  stream=wp.stream_from_torch(torch.cuda.current_stream()))
    first, stream = make_kernel(1.0), torch.cuda.Stream()
    with torch.cuda.stream(stream):
        launch(first)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch(first)
    launch(make_kernel(2.0))  # reloads the module
    torch.cuda.synchronize()
    import gc; gc.collect()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    print(float(x[0]))
""")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graph_replay_survives_a_module_reload(tmp_path):
    script = tmp_path / "reload_during_graph_use.py"  # Warp reads kernel source from a file
    script.write_text(SCRIPT)
    result = subprocess.run(
        [sys.executable, str(script), str(Path(__file__).parents[1])], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "6.0"  # eager 1 + eager 2 + three replays
