# Packaging Spec Examples

## When to Read

- 执行 Required Workflow 步骤 4（创建 packaging spec）时
- 需要参考 PI0.5 或 SmolVLA 的 spec JSON 结构时

Use one JSON spec containing only source paths. Relative paths resolve from the spec directory.

## PI0.5 Example

```json
{
  "vision": {
    "artifact": "/compiler/pi05/vision.hmm",
    "abi": "/compiler/pi05/vision/model.json"
  },
  "embedding": "/compiler/pi05/embedding.pt",
  "roles": {
    "prefill": {
      "artifact": "/compiler/pi05/prefill.hmm",
      "abi": "/compiler/pi05/prefill/model.json"
    },
    "action_in_proj": {
      "artifact": "/compiler/pi05/action_in_proj.hmm",
      "abi": "/compiler/pi05/action_in_proj/model.json"
    },
    "time_mlp": {
      "artifact": "/compiler/pi05/time_mlp.hmm",
      "abi": "/compiler/pi05/time_mlp/model.json"
    },
    "decode": {
      "artifact": "/compiler/pi05/decode.hmm",
      "abi": "/compiler/pi05/decode/model.json"
    },
    "action_out_proj": {
      "artifact": "/compiler/pi05/action_out_proj.hmm",
      "abi": "/compiler/pi05/action_out_proj/model.json"
    }
  },
  "vision_layout": "NCHW"
}
```

## SmolVLA Example

```json
{
  "vision": {
    "artifact": "/compiler/smolvla/vision.hmm",
    "abi": "/compiler/smolvla/vision/model.json"
  },
  "embedding": "/compiler/smolvla/token_embedding.pt",
  "state_projection": "/compiler/smolvla/state_projection.pt",
  "roles": {
    "prefill": {
      "artifact": "/compiler/smolvla/prefill.hmm",
      "abi": "/compiler/smolvla/prefill/model.json"
    },
    "action": {
      "artifact": "/compiler/smolvla/action.hmm",
      "abi": "/compiler/smolvla/action/model.json"
    }
  },
  "vision_layout": "NCHW"
}
```
