"""Test each model in llm_models.yaml with a minimal prompt.

Usage:
    python test/test_llm_client.py                      # test all models
    python test/test_llm_client.py qwen-max kimi        # test specific models

Requires the corresponding API key env vars to be set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from llm_client import _DEFAULT_CONFIG, call_llm


def test_models(target_models: list[str] | None = None) -> bool:
    with open(_DEFAULT_CONFIG) as f:
        config = yaml.safe_load(f).get("llm_client", {})

    all_models = config.get("models", {})
    models_to_test = target_models or list(all_models)
    unknown = [m for m in models_to_test if m not in all_models]
    if unknown:
        print(f"Unknown models (not in yaml): {unknown}")
        return False

    print(f"Testing {len(models_to_test)} model(s)...\n")

    passed = 0
    for name in models_to_test:
        print(f"  [{name}] ", end="", flush=True)
        resp = call_llm(model=name, prompt='Reply with exactly one word: "ok"')

        if resp["status_code"] == 200:
            ms = resp["timing"]["request_ms"]
            snippet = resp["content"][:60].replace("\n", " ")
            print(f"OK  {ms}ms  {snippet!r}")
            passed += 1
        else:
            print(f"FAIL {resp['status_code']}  {resp['message']}")

    print(f"\n{passed}/{len(models_to_test)} passed")
    return passed == len(models_to_test)


if __name__ == "__main__":
    targets = sys.argv[1:] or None
    ok = test_models(targets)
    sys.exit(0 if ok else 1)
