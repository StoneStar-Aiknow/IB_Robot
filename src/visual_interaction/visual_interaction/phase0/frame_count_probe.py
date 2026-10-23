"""Does the model read motion across frames, and are four frames worth it?

Run on the board, against frames a previous demo run saved. Three configurations
answer the same question about the same pictures:

``4frames_motion``  four frames, asked what *changed* between them
``2frames_motion``  first and last frame only, same question
``4frames_static``  four frames, asked what the person *is doing* (the control)

A question a single frame could answer proves nothing about time, so the motion
wording asks explicitly for a before/after comparison. Greedy decoding, no
sampling and no retries, so a rerun gives the same text.

Result of the first run (12 rounds, 2026-10-21): every one of the 24 motion
answers said "no change", and the motion wording also lost detail the static
wording got right — so the first version claims no cross-frame understanding.
Two frames matched four on quality while costing 526 fewer prompt tokens.
See docs/x2_visual_interaction_scope_change_zh.md.

Usage:
    python3 frame_count_probe.py          # needs CANN env; loads three OMs
"""

import json
import os
import sys
import time

import numpy as np

ROOT = "/root/vlx_internvl_310p"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "live"))

IMG_TOKENS = 256
SYSTEM = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"

MOTION_Q = (
    "以上是摄像头按时间顺序拍到的 {n} 帧画面（第1帧最早，第{n}帧最晚）。"
    "只看离镜头最近的那个人：对比这几帧，这个人的动作或姿态发生了什么变化？"
    "如果几帧之间没有变化，就回答：没有变化"
)
STATIC_Q = (
    "以上是摄像头最近 2 秒内按时间顺序拍到的 {n} 帧画面。"
    "只看离镜头最近的那个人：这个人正在做什么动作、手势，是什么表情？"
    "用一句话回答，不要描述桌子和背景。如果没有人，只回答：无人"
)


def chat(q):
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def prompt(n, template):
    q = "".join(f"Frame{i + 1}: <image>\n" for i in range(n)) + template.format(n=n)
    return chat(q).replace("<image>", "<img>" + "<IMG_CONTEXT>" * IMG_TOKENS + "</img>")


def _read_frames(directory: str, count: int = 4) -> list[bytes]:
    jpegs = []
    for index in range(count):
        with open(os.path.join(directory, f"{index}.jpg"), "rb") as stream:
            jpegs.append(stream.read())
    return jpegs


def main():
    import live_chat
    from llm_host import Generator
    from qwen_detok import Detokenizer
    from qwen_tok import Tokenizer
    from run_llm_om import OM, om_backend, session

    live_chat.quiet_native_output(os.devnull)
    tok, detok = Tokenizer(f"{ROOT}/tokenizer"), Detokenizer(f"{ROOT}/tokenizer")
    img_ctx, eos = tok.added["<IMG_CONTEXT>"], tok.added["<|im_end|>"]

    vis = om_backend(session(OM["internvl3_1b_vision_b4_fp16"]), ["vit_embeds"])
    names = ["logits", "new_key", "new_value"]
    embed = np.load(f"{ROOT}/onnx/embed_tokens_fp32.npy", mmap_mode="r")
    gen = Generator(
        embed,
        om_backend(session(OM["llm_prefill_fp16"]), names),
        om_backend(session(OM["llm_decode_fp16"]), names),
        1280,
        1344,
        img_ctx,
        eos,
    )

    frames_root = f"{ROOT}/out/live/20261014_161315/frames"
    rounds = sorted(os.listdir(frames_root))[:12]
    configs = [("4frames_motion", 4, MOTION_Q), ("2frames_motion", 2, MOTION_Q), ("4frames_static", 4, STATIC_Q)]
    ids = {name: np.array(tok.encode(prompt(n, t))) for name, n, t in configs}
    for name, n, _ in configs:
        print(f"{name}: {len(ids[name])} prompt token ({n} frames)", flush=True)

    # The vision OM is compiled for a batch of 4, so a 2-frame window is padded
    # and only the first two slots are wired into the prompt.
    out = []
    for rd in rounds:
        d = os.path.join(frames_root, rd)
        jpegs = _read_frames(d)
        rec = {"round": rd}
        for name, n, _ in configs:
            picked = jpegs if n == 4 else [jpegs[0], jpegs[3]]
            pv = np.zeros((4, 3, 448, 448), np.float32)
            pv[:n] = live_chat.preprocess(picked)
            vit = vis({"pixel_values": pv})[0].reshape(4, IMG_TOKENS, -1)[:n]
            t = time.time()
            r = gen.run(ids[name], vit, max_new_tokens=64, keep_logits=False)
            rec[name] = {"text": detok.decode(r["tokens"]).strip(), "s": round(time.time() - t, 2)}
        out.append(rec)
        print(f"[{rd}] 4f-motion: {rec['4frames_motion']['text']}", flush=True)
        print(f"      2f-motion: {rec['2frames_motion']['text']}", flush=True)
        print(f"      4f-static: {rec['4frames_static']['text']}", flush=True)
    with open("/tmp/vlprobe_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote /tmp/vlprobe_result.json")


if __name__ == "__main__":
    main()
