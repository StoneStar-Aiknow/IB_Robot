# llm_client_service

无状态的云端大模型单次调用接口。给定输入 → 调用云端模型 → 返回统一格式的 JSON 结果。

## 文件说明

```
llm_client_service/
├── llm_client.py        # 唯一的接口文件，import 这个就够了
├── llm_models.yaml      # 命名模型配置表，新增厂家/模型在这里加
├── README.md
└── test/
    ├── test_llm_client.py   # 验证各模型是否跑通
    └── demo.py              # 4 种调用模式的参考示例
```

---

## 快速上手

### 1. 配置 API Key

Key 永远通过**环境变量**传入，不写在 yaml 里。根据你要用的模型，设置对应的环境变量：

```bash
export ALIYUN_API_KEY="sk-..."      # 阿里百炼（qwen3-vl-* / deepseek-v4-pro）
export KIMICODE_API_KEY="..."       # Kimi
export DEEPSEEK_API_KEY="..."       # DeepSeek（直连）
```

### 2. 在代码里调用

```python
import sys
sys.path.insert(0, "/path/to/src/embodied_agent/llm_client_service")

from llm_client import call_llm

resp = call_llm(model="qwen3-vl-32b-thinking", prompt="桌上有什么物体？")
if resp["status_code"] == 200:
    print(resp["content"])
else:
    print(f"Error {resp['status_code']}: {resp['message']}")
```

### 3. 运行连通性测试

```bash
cd src/embodied_agent/llm_client_service

# 测所有配置的模型
python test/test_llm_client.py

# 只测指定模型
python test/test_llm_client.py qwen-max kimi
```

输出示例：
```
Testing 4 model(s)...

  [qwen3-vl-235b-a22b-thinking] OK  843ms  'ok'
  [qwen3-vl-32b-thinking]       OK  1204ms  'ok'
  [deepseek-v4-pro]             OK  2341ms  'ok'
  [deepseek]                    FAIL 401  API key env var not set: DEEPSEEK_API_KEY

3/4 passed
```

---

## 接口说明

### `call_llm(...)` 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `model` | str | ✅ | 模型名，必须是 `llm_models.yaml` 里定义的 key |
| `prompt` | str | — | 用户消息。单独使用时作为唯一 user 消息；与 `messages` 同时传入时，追加为最新一条 user 消息（多轮对话场景） |
| `messages` | list[dict] | — | 完整的对话历史（OpenAI message 格式），见[多轮对话](#多轮对话) |
| `system` | str | — | System prompt，始终作为第一条消息插入 |
| `image_path` | str | — | 本地图片路径，自动编码附到最后一条 user 消息。模型不支持多模态时自动降级为纯文本 |
| `tools` | list[dict] | — | OpenAI 格式的工具/函数定义，原样透传给云端，见 [Tools 调用](#tools-调用) |
| `force_json` | bool | — | 要求模型只输出 JSON（传了 `tools` 时此参数无效） |
| `config_path` | str/Path | — | 覆盖默认的 `llm_models.yaml` 路径 |

### 返回值（统一 JSON dict）

| 字段 | 类型 | 说明 |
|------|------|------|
| `status_code` | int | 200 成功，其他见[状态码](#状态码) |
| `message` | str | 人类可读的状态说明 |
| `model` | str | 本次使用的命名模型 key |
| `provider` | str | 固定为 `openai_compatible` |
| `content` | str | 模型输出的文本内容 |
| `tool_calls` | list | 模型请求调用的工具列表（无则为空列表），见 [Tools 调用](#tools-调用) |
| `reasoning` | str | 模型的思考过程（DeepSeek R 系列等有推理能力的模型才有，其余为空） |
| `usage` | dict | Token 用量：`prompt_tokens`、`completion_tokens`、`total_tokens` |
| `timing` | dict | 耗时：`request_ms`（发出请求到收到响应）、`total_ms`（含参数处理） |
| `raw` | dict | 云端返回的原始响应，调试用 |

### 状态码

| 状态码 | 含义 | 常见原因 |
|--------|------|---------|
| 200 | 成功 | — |
| 400 | 请求参数错误 | 没有 user 消息、图片文件不存在 |
| 401 | 鉴权失败 | API key 环境变量未设置，或 key 被厂家拒绝 |
| 404 | 模型未找到 | `model` 参数不在 `llm_models.yaml` 中 |
| 408 | 请求超时 | 超过 `default_timeout_sec` |
| 429 | 被限流 | 厂家 QPS 限制 |
| 502 | 上游错误 | 厂家返回 5xx、响应体不是合法 JSON、choices 为空 |
| 503 | 网络错误 | DNS 解析失败、连接被拒 |
| 500 | 内部错误 | 配置文件加载失败等未预期异常 |

---

## 使用场景示例

### 纯文本问答

```python
resp = call_llm(
    model="qwen3-vl-32b-thinking",
    system="你是一个机器人任务规划助手，只输出 JSON。",
    prompt="把桌上的红色方块放到托盘里",
)
print(resp["content"])
```

### 带图像的多模态请求

```python
resp = call_llm(
    model="qwen3-vl-235b-a22b-thinking",  # 需要 multimodal: true 的模型
    prompt="描述一下图中的场景",
    image_path="/tmp/camera_snapshot.jpg",
)
```

模型不支持图像时（`multimodal: false`），图片会被自动忽略，`message` 字段会标注 `image ignored: model is text-only`，请求仍会正常发出。

### 要求输出 JSON

```python
resp = call_llm(
    model="deepseek-v4-pro",
    system="只输出合法 JSON，不要有多余文字。",
    prompt="给我一个包含 name 和 age 字段的 JSON 示例",
    force_json=True,
)
import json
data = json.loads(resp["content"])
```

### Tools 调用（让模型选择技能）

将 skill 定义为 OpenAI 格式的 function，传入 `tools`。接口原样透传给云端，不解析也不执行，调用方自行处理 `tool_calls`。

```python
resp = call_llm(
    model="qwen3-vl-32b-thinking",
    system="你是机器人规划器，根据用户指令选择合适的技能执行。",
    prompt="回到初始位置",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "move_to_named_pose",
                "description": "Move the robot arm to a named pose",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pose_name": {
                            "type": "string",
                            "description": "Target pose, e.g. home, zero, observe_table",
                        }
                    },
                    "required": ["pose_name"],
                },
            },
        },
    ],
)

# 模型直接回答时 tool_calls 为空，content 有内容
# 模型决定调用工具时 tool_calls 非空，content 可能为空
if resp["tool_calls"]:
    for call in resp["tool_calls"]:
        fn = call["function"]
        args = json.loads(fn["arguments"])
        print(f"Model wants to call: {fn['name']}({args})")
```

### 多轮对话

接口本身无状态，**调用方负责维护对话历史**，每轮把完整历史通过 `messages` 传入。

核心模式：每次调用后把本轮的 user/assistant 消息追加到 history，下一轮连同 history 一起传入。

```python
history = []

def chat(user_message: str) -> str:
    resp = call_llm(model="qwen3-vl-32b-thinking", system="你是机器人助手。", messages=history, prompt=user_message)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": resp["content"]})
    return resp["content"]

print(chat("桌上有什么？"))
print(chat("把红色的那个放到托盘"))   # 模型记得上文
print(chat("完成了吗？"))
```

> 注意：传入 `messages` 时，`prompt` 参数会被追加为新的一条 user 消息。如果 `messages` 里最后一条已经是你要说的话，直接不传 `prompt` 即可。
>
> 可运行示例参见 [`test/demo.py`](test/demo.py) 第 4 节。

---

## 添加新模型

在 `llm_models.yaml` 的 `models:` 下新增一条即可，不需要改任何代码：

```yaml
# 示例：添加 OpenAI GPT-4o
gpt-4o:
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  model_id: gpt-4o
  multimodal: true
```

所有 OpenAI 兼容接口（`/chat/completions`）的厂家都可以用这种方式接入。

---

## 注意事项

- **API Key 安全**：`api_key_env` 填的是环境变量的**名字**，不是 key 本身。Key 只存在于运行时环境变量中，不会出现在代码或配置文件里。
- **超时**：默认 60 秒。大模型推理可能较慢，按需在 yaml 里调整 `default_timeout_sec`。
- **图像大小**：图片会被完整读入内存编码为 base64，大图会增加请求体积和耗时，建议提前压缩（参考现有 `vlm_api_client.py` 里的 `jpeg_quality` 处理方式）。
