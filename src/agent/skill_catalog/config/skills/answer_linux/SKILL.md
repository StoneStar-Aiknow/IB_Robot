# answer_linux

Say with a recorded reply that the robot is developed on openEuler, a Linux distribution.

- executor: `runtime_play_audio` (delegated); the catalog binds the logical clip name `linux_based` (recordings `Q20` in `commissioning/audio/summit_clips.yaml`).
- recordings: 我是基于 openEuler 开发的，openEuler 是基于 Linux 的发行版哦！ / 嗯嗯，我基于 openEuler 开发，它是一个 Linux 发行版呀～
- clip only, no gesture: a short answer with a gesture would leave the arms moving after the words.
- list the skill in `voice_bridge.silent_skills`: the clip is the reply.
- aliases: 你是基于Linux的吗 / 你是Linux吗 / 你用的是Linux吗 / openEuler是Linux吗.
