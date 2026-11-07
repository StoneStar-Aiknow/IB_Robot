# Test scope and consolidation

The suite keeps one integration test per boundary and focused tests for failure
semantics. Removed duplication is tracked here so it is not regenerated later.

| Removed or reduced | Retained coverage |
| --- | --- |
| `test_raise_return_hands_mock.py`: duplicate generated-pose/hand transport assembly | `test_fixed_poses.py` covers real fixed execution, both hand models and HOME return. `test_core.py` / `test_spec.py` cover generated composition and planning; Gateway routing remains in `test_task_pose_gateway_mock.py`. |
| `test_voice_memory_mock.py`: repeated scripted phrases, fake voice playback and memory scenarios | `test_agent_memory_mock.py` retains one real Agent/Composer/SQLite consent-and-recall smoke test. `test_composer_memory.py` covers lifecycle and consent; Agent `test_pose_preview.py` / `test_voice_bridge.py` cover semantic decisions, original wording and speech routing. Scripted phrase synonyms did not test the production language model. |
| `test_task_pose_hands.py`: second real-restorer transport fake | `test_fixed_poses.py` covers successful real dual-hand restoration; `aimdk_grasp/test/test_hand_restore.py` covers cancellation, release and partial-output failures. Runner budget, approvals and partial-completion tests remain local. |
| Fixed-executor policy/gesture cross product | Keep named `grasp_ready`, `arms_down` returning to `arms_rest`, left/right passive-arm rest and dual-hand opening. Four policy combinations and full-path refusals remain in `test_path_validation.py`; widened finger-envelope refusals remain in geometry/compiler tests. Every seed's composition and aliases remain tested in `test_seeds.py`. |
| Repeated per-phase Gateway failure injection | Keep preflight failure, partial hand completion, confirmation failure, cancellation and mode changes at the Gateway boundary. Stale feedback, timeout, digest and earlier opening failures remain in runner/restorer unit tests. |
| Agent message length × status cross product | Keep Unicode limits at 300/301, a 2208-character failed Program, durable failure without quarantine and existing terminal-classification tests. |
| Agent/SQLite request wiring repeated in read-only and writable recall tests | Share the `agent_session` fixture; keep both recall boundaries and their model-call assertions. |
| `test_compile_library.py`: repeated HOME setup and online rejection | `test_home_seed_compiler.py` keeps the strict online refusal and explicit off-policy scope check, with shared model restoration. |
| Separate successful Gateway preparation tests with and without hands | Parameterize the same success assembly; retain hand ordering, completed stages, budgets and failure cases. |

No production checks are disabled by this consolidation. Physical acceptance is
separate from Mock acceptance. Run through `colcon test --merge-install`; set
`IBROBOT_FIXED_LIBRARY` to the expanded candidate artifact for fixed-mode cases.
