# Execution launch baseline

`execution_direct_baseline.json` captures the direct execution graph from commit
`acea31b54732d4bf7769f0cb5a285874cf5cd5b6`.
Both `launch_builders/execution.py` and `inference_config.py` were loaded from that commit.
Both files are byte-identical in the PR base `c8be174e46156a94219999a8dc216336f1f19af5`.
The input scenarios are defined in `_direct_baseline_config` in `test_execution_scheduler_branch.py`.

Capture used `_launch_snapshot` without starting ROS processes. Only temporary paths are replaced with `<tmp>`;
`prepare_lerobot_env` returns the fixed `IBROBOT_TEST_ENV=preserved` environment, and the host
`OVERRIDE_LAUNCH_PROCESS_OUTPUT` override is removed. Inherited process environment remains `null`.
Benchmark exit handlers are recorded in action order with the indices of the nodes whose exit events they match.

Tests read this fixed fixture without requiring Git history. Do not regenerate it from the implementation under test:
an intentional contract change requires reviewing the old/new behavior and updating the documented baseline.
Coverage is the generated execution graph for equal effective targets and clocks, not process execution.

The `mock_default_clock` and `sdk_default_clock` scenarios were captured separately from the complete
`robot_config` package at `c8be174e46156a94219999a8dc216336f1f19af5`, recorded as
`default_clock_baseline_commit`. They pass `use_sim=True` and omit `use_sim_time`, so their fixed snapshots
also protect the standalone builder's pre-migration clock default. Both full execution and inference-only
builders are compared against these snapshots with scheduling absent and explicitly disabled.

`preparation_direct_baseline.json` captures the complete preparation result fields listed in each scenario
from the PR base `c8be174e46156a94219999a8dc216336f1f19af5`. The entire `robot_config` Python package was
loaded from a Git archive of that commit, using the installed ROS dependencies. Only config-path lookup and
host resource validation were replaced, as in the `business` fixture of `test_launch_preparation.py`.
The registry has no business builders. Inputs cover target defaults, YAML `use_sim`, CLI target/embodiment/platform
selection, conflicting CLI arguments and benchmark. Expected results retain the prepared configuration and clock.
The test checks both absent and disabled scheduling; the only omitted output field is the newly supplied
`runtime_scheduler` block itself. Source dictionaries must remain unchanged.
