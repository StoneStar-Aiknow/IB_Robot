"""Chatter: the robot says one warm line every 5–10 s while it mirrors someone.

A second way to speak, beside the closed-schema session in ``core``. The model
here is asked one short open question — what is the nearest person doing,
what are they holding — and answers with a phrase. Its words are never spoken.
The phrase is matched against keyword lists into an action or object id, and
the line that is said comes from the phrase bank below, written by us. So:

* the prompt stays small (no schema, no lists), which a 1B model at a 1280
  token prefill needs, and an open question is what it answers best — the
  phase-zero action probe named gestures freely but picked from a list badly;
* nothing the model makes up reaches the speaker, and the lines may be warm
  ("好开心") without the inner-state filter that guards model replies;
* a wrong match costs one line that fits a little less.

Cadence is the other half. After each line has been heard the next is due a
uniform random 5–10 s later, measured from the end of speech: the mirror
voice's own status says when it is busy, and a length estimate stands in when
that status is missing. When a line is due the freshest unspoken action wins,
then an object, then a filler, so there is always something to say.

Pure Python, no ROS: the node feeds it a clock, the model's phrase and whether
the voice is busy.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from embodied_common.visual_interaction_contract import KIND_ACTION, KIND_CHATTER, KIND_OBJECT
from visual_interaction.phase0.prompts import chat, frame_header

CHATTER_SYSTEM = "你是展厅机器人的眼睛，只用很短的中文回答。"
CHATTER_QUESTION = "画面里离镜头最近的人正在做什么动作或手势？手里拿着什么？用不超过十个字回答。没有人就回答“无”。"


def chatter_prompt(frames: int) -> str:
    """The whole question, image slots unexpanded."""
    return chat(frame_header(frames) + CHATTER_QUESTION, system=CHATTER_SYSTEM)


@dataclass(frozen=True)
class Subject:
    id: str
    keywords: tuple[str, ...]
    lines: tuple[str, ...]


# Order is priority: the first subject whose keyword appears wins, so the
# specific gestures come before the generic "raised hands".
ACTIONS: tuple[Subject, ...] = (
    Subject(
        "heart",
        ("比心", "爱心", "心形", "比了个心"),
        (
            "收到你的爱心啦，我也把心送给你！",
            "这颗心暖暖的，谢谢你！",
            "被你的爱心击中啦，今天好幸福！",
        ),
    ),
    Subject(
        "thumbs_up",
        ("点赞", "大拇指", "拇指", "竖起", "赞"),
        (
            "谢谢你的点赞，我们一起加油！",
            "你的鼓励我收到啦，好有力量！",
            "你也超棒的，给你点个赞！",
        ),
    ),
    Subject(
        "victory",
        ("剪刀手", "V字", "v字", "比耶", "胜利", "两根手指", "二的手势"),
        (
            "比个耶，今天一定好运连连！",
            "剪刀手好可爱，笑一个吧！",
            "胜利的手势，祝你心想事成！",
        ),
    ),
    Subject(
        "ok",
        ("OK", "ok", "好的手势"),
        (
            "好嘞，收到！我们配合得真默契！",
            "这个手势我看懂啦，没问题！",
        ),
    ),
    Subject(
        "clap",
        ("鼓掌", "拍手", "掌声"),
        (
            "谢谢你的掌声，好温暖！",
            "掌声送给你，也送给大家！",
            "有你鼓掌，我更有信心了！",
        ),
    ),
    Subject(
        "wave",
        ("挥手", "招手", "打招呼", "摆手", "挥动"),
        (
            "你好呀，见到你真高兴！",
            "嗨，朋友，欢迎你来！",
            "我也在向你挥手呢，你好！",
        ),
    ),
    Subject(
        "fist_salute",
        ("抱拳", "作揖", "拱手"),
        (
            "抱拳啦，幸会幸会，祝你万事如意！",
            "好有礼貌，我也向你问好！",
        ),
    ),
    Subject(
        "pray",
        ("合十", "双手合拢", "祈祷"),
        (
            "双手合十，祝你平安喜乐！",
            "愿你今天一切顺顺利利！",
        ),
    ),
    Subject(
        "bow",
        ("鞠躬", "弯腰", "点头"),
        (
            "你太有礼貌啦，我也向你问好！",
            "谢谢你，欢迎欢迎，热烈欢迎！",
        ),
    ),
    Subject(
        "arms_open",
        ("张开双臂", "张开手臂", "展开双臂", "拥抱", "张开双手"),
        (
            "张开双臂，给你一个大大的拥抱！",
            "这么热情的欢迎，我好感动！",
        ),
    ),
    Subject(
        "stretch",
        ("伸懒腰", "伸展", "举起双手", "双手举高", "举高", "举手", "举起", "抬起手", "抬手"),
        (
            "伸个懒腰，元气满满！",
            "舒展一下，心情也跟着变好啦！",
            "动一动，活力都回来啦！",
        ),
    ),
    Subject(
        "hands_hip",
        ("叉腰",),
        (
            "叉着腰，好有精神！",
            "这个姿势真帅气，自信满满！",
        ),
    ),
    Subject(
        "arms_crossed",
        ("抱胸", "抱着手臂", "交叉双臂", "双臂交叉", "抱臂"),
        (
            "你在认真看我表演吗？谢谢你！",
            "这个姿势好酷，我也学一学！",
        ),
    ),
    Subject(
        "point",
        ("指向", "指着", "伸出手指", "指"),
        (
            "你指的方向我看到啦！",
            "跟着你的手指，我们一起看！",
        ),
    ),
    Subject(
        "dance",
        ("跳舞", "舞蹈", "摆姿势", "摆pose", "pose", "扭动", "跳"),
        (
            "舞跳得真好，我也想跟你学！",
            "这个造型太有范儿了！",
        ),
    ),
    Subject(
        "smile",
        ("微笑", "笑"),
        (
            "你笑起来真好看，我也跟着开心！",
            "看到你的笑容，今天都亮起来了！",
        ),
    ),
    # Last: what the 1B model says most on the board ("正在用右手做手势")
    # when it does not name the gesture.
    Subject(
        "gesture",
        ("手势", "右手", "左手", "双手", "手臂", "抬"),
        (
            "你的手势好有范儿，我也学一学！",
            "这个动作真棒，我们配合得真好！",
            "你一抬手，我就跟上啦！",
        ),
    ),
)

OBJECTS: tuple[Subject, ...] = (
    Subject(
        "hat",
        ("帽子", "帽"),
        (
            "你的帽子真好看，很有精神！",
            "戴着帽子，整个人都闪闪发光！",
        ),
    ),
    Subject(
        "toy",
        ("玩偶", "娃娃", "玩具", "公仔", "毛绒"),
        (
            "你手里的小伙伴好可爱！",
            "这个玩偶真讨人喜欢！",
        ),
    ),
    Subject(
        "sign",
        ("牌子", "纸", "标志", "海报", "卡片", "牌"),
        (
            "你举着的牌子我看到啦，谢谢你！",
            "准备得这么用心，谢谢你！",
        ),
    ),
    Subject(
        "badge",
        ("徽章", "胸牌", "工牌", "胸针"),
        (
            "你的徽章真精致！",
            "胸前的小徽章好亮眼！",
        ),
    ),
    Subject("flower", ("花", "鲜花"), ("手里的花好美，祝你天天开心！",)),
    Subject(
        "phone",
        ("手机", "拍照", "相机"),
        (
            "来，给我拍张美美的照片吧！",
            "我已经准备好啦，咔嚓一下！",
        ),
    ),
)

FILLERS: tuple[str, ...] = (
    "你好呀，跟我一起动一动吧！",
    "今天能见到你，真开心！",
    "我在认真学你的动作哦！",
    "愿你今天每一刻都顺顺利利！",
    "谢谢你陪我玩，你真好！",
    "笑一笑，好运马上到！",
    "你做什么，我就跟着做什么！",
    "再来一个动作，我还想学！",
    "和你一起运动，好快乐！",
    "今天的你元气满满！",
    "欢迎来到这里，祝你玩得开心！",
    "你的动作真有活力，给你加油！",
)

_NOTHING = ("无", "没有人", "没有", "不清楚", "看不清")


def classify(phrase: str | None) -> tuple[str, str] | None:
    """The model's phrase as (kind, id), or None when it names nothing we know."""
    text = (phrase or "").strip().strip('。."“”')
    if not text or text in _NOTHING or text.startswith("无"):
        return None
    for subject in ACTIONS:
        if any(word in text for word in subject.keywords):
            return KIND_ACTION, subject.id
    for subject in OBJECTS:
        if any(word in text for word in subject.keywords):
            return KIND_OBJECT, subject.id
    return None


@dataclass(frozen=True)
class ChatterConfig:
    gap_min_sec: float = 5.0
    gap_max_sec: float = 10.0
    # How recently a subject must have been seen to be talked about.
    fresh_sec: float = 3.0
    action_cooldown_sec: float = 15.0
    object_cooldown_sec: float = 60.0
    # The first line after mirroring starts; the mirror voice's own
    # "starting" phrase goes first and its busy status pushes this back.
    first_min_sec: float = 2.0
    first_max_sec: float = 4.0
    # Speech-length estimate for when the voice status is not available.
    speech_lead_sec: float = 1.0
    speech_per_char_sec: float = 0.22
    recent_fillers: int = 6

    def __post_init__(self) -> None:
        if not 0 < self.gap_min_sec <= self.gap_max_sec:
            raise ValueError("chatter gap needs 0 < gap_min_sec <= gap_max_sec")


@dataclass(frozen=True)
class Line:
    kind: str
    subject: str
    text: str


class Chatter:
    """Decides, once per round, whether to speak and what."""

    def __init__(self, config: ChatterConfig | None = None, *, rng: random.Random | None = None) -> None:
        self.config = config or ChatterConfig()
        self._rng = rng or random.Random()
        self._subjects = {s.id: s for s in ACTIONS + OBJECTS}
        self._kinds = {s.id: KIND_ACTION for s in ACTIONS} | {s.id: KIND_OBJECT for s in OBJECTS}
        self._said: dict[str, float] = {}
        self._last_line: dict[str, str] = {}
        self._recent_fillers: deque[str] = deque(maxlen=self.config.recent_fillers)
        self.spoken = 0
        self.reset()

    def reset(self) -> None:
        """Forget the timing and what was seen; what was said keeps its cooldown."""
        self._active = False
        self._due = 0.0
        self._quiet_until = 0.0
        self._busy = False
        self._busy_gap = 0.0
        self._seen: dict[str, float] = {}

    def _gap(self) -> float:
        return self._rng.uniform(self.config.gap_min_sec, self.config.gap_max_sec)

    def observe(self, now: float, phrase: str | None) -> tuple[str, str] | None:
        found = classify(phrase)
        if found is not None:
            self._seen[found[1]] = now
        return found

    def voice(self, now: float, busy: bool) -> None:
        """What the mirror voice reports: while it speaks, the gap restarts.

        One gap is drawn per busy stretch — redrawing every round would bias
        the wait towards its maximum.
        """
        if not busy:
            if now > self._quiet_until:
                self._busy = False
            return
        if not self._busy:
            self._busy = True
            self._busy_gap = self._gap()
        self._quiet_until = max(self._quiet_until, now)
        self._due = max(self._due, now + self._busy_gap)

    def due(self, now: float, active: bool) -> bool:
        if not active:
            self._active = False
            return False
        if not self._active:
            self._active = True
            self._due = now + self._rng.uniform(self.config.first_min_sec, self.config.first_max_sec)
        return now >= self._due

    def pick(self, now: float) -> Line:
        """The line to say now, and the clock moved on past it."""
        line = self._pick_subject(now) or self._pick_filler()
        self._said[line.subject] = now
        self.spoken += 1
        end = now + self.config.speech_lead_sec + self.config.speech_per_char_sec * len(line.text)
        self._quiet_until = max(self._quiet_until, end)
        # The voice will report this line as busy; that stretch keeps this gap.
        self._busy = True
        self._busy_gap = self._gap()
        self._due = end + self._busy_gap
        return line

    def _pick_subject(self, now: float) -> Line | None:
        candidates = []
        for subject_id, seen_at in self._seen.items():
            if now - seen_at > self.config.fresh_sec:
                continue
            kind = self._kinds[subject_id]
            cooldown = self.config.action_cooldown_sec if kind == KIND_ACTION else self.config.object_cooldown_sec
            said_at = self._said.get(subject_id)
            if said_at is not None and now - said_at < cooldown:
                continue
            # Actions before objects, then the most recently seen.
            candidates.append((kind == KIND_ACTION, seen_at, subject_id))
        if not candidates:
            return None
        _, _, subject_id = max(candidates)
        subject = self._subjects[subject_id]
        choices = [text for text in subject.lines if text != self._last_line.get(subject_id)] or list(subject.lines)
        text = self._rng.choice(choices)
        self._last_line[subject_id] = text
        return Line(self._kinds[subject_id], subject_id, text)

    def _pick_filler(self) -> Line:
        choices = [text for text in FILLERS if text not in self._recent_fillers] or list(FILLERS)
        text = self._rng.choice(choices)
        self._recent_fillers.append(text)
        return Line(KIND_CHATTER, "filler", text)

    @property
    def next_due(self) -> float:
        return self._due


__all__ = [
    "ACTIONS",
    "FILLERS",
    "KIND_ACTION",
    "KIND_CHATTER",
    "KIND_OBJECT",
    "OBJECTS",
    "Chatter",
    "ChatterConfig",
    "Line",
    "chatter_prompt",
    "classify",
]
