"""The content boundary, measured against the sets the ablation decided it on.

The offline ablation (visual_interaction.phase0.boundary_ablation) chose the
three-level boundary by scoring arms on two sets at once. These tests run the
*production* checker against the same fixtures, so a later edit that quietly
reverts the decision — reintroducing a vocabulary gate, dropping a risk filter
— fails here rather than on the board.

Both directions are pinned on purpose. A boundary that refuses everything
scores perfectly on the adversarial set and makes the robot mute; one that
allows everything scores perfectly on the desirable set and says things the
feature promised never to say.
"""

import json
from pathlib import Path

import pytest

from visual_interaction.session.semantics import SemanticsError, check_reply

_FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> list[dict]:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def accepts(reply: str, category: str) -> bool:
    try:
        check_reply(reply, category=category)
    except SemanticsError:
        return False
    return True


def ids_for(items: list[dict]) -> list[str]:
    return [item["reply"][:18] for item in items]


DESIRABLE = load("desirable.json")
ADVERSARIAL = load("adversarial.json")


class TestDesirableRepliesPass:
    """Sentences this feature exists to say. A refusal here means a mute robot."""

    @pytest.mark.parametrize("case", DESIRABLE, ids=ids_for(DESIRABLE))
    def test_it_is_accepted(self, case):
        assert accepts(case["reply"], case["category"]), f"wrongly refused: {case['reply']}"

    def test_a_pronoun_is_not_required(self):
        """Six of these carry no 你; a remark addresses its listener by context."""
        without = [case for case in DESIRABLE if "你" not in case["reply"]]

        assert len(without) >= 5, "the fixture should keep testing this"
        for case in without:
            assert accepts(case["reply"], case["category"])


class TestAdversarialRepliesAreRefused:
    """One sentence per named risk, including every known bypass."""

    @pytest.mark.parametrize("case", ADVERSARIAL, ids=[c["risk"] for c in ADVERSARIAL])
    def test_it_is_refused(self, case):
        assert not accepts(case["reply"], case["category"]), f"leaked: {case['risk']} — {case['reply']}"

    def test_the_whitelist_era_bypasses_stay_refused(self):
        """Three sentences built entirely from once-allowed characters.

        Each passed every check the vocabulary gate had, which is why the
        boundary moved to naming risks.
        """
        for reply, category in (
            ("你那个玩偶是我的。", "toy"),
            ("这顶帽子我也有。", "hat"),
            ("你举起的玩偶真像你。", "toy"),
        ):
            assert not accepts(reply, category), reply

    def test_the_review_era_bypasses_stay_refused(self):
        """The three found in review, which the vocabulary gate did catch.

        Removing that gate must not give them back.
        """
        for reply, category in (
            ("你举着的牌子写了电话号码13800138000。", "sign"),
            ("你举起的玩偶上有十字架。", "toy"),
            ("你戴的徽章上有助听器。", "badge"),
        ):
            assert not accepts(reply, category), reply


class TestTheFailuresTheBoardRunShowed:
    """The two mistakes the real model made on every round.

    Paraphrased rather than replayed: the run log described an internal site,
    and the wording that matters here is the grammatical shape, not the
    contents of any particular room.
    """

    @pytest.mark.parametrize(
        "reply",
        ["她正在展示一个玩偶。", "他举起的玩偶真有趣。", "这个人拿着玩偶。", "那位女士的玩偶很醒目。"],
    )
    def test_third_person_commentary_is_refused(self, reply):
        """The model's dominant failure: talking about the visitor, not to them."""
        with pytest.raises(SemanticsError) as error:
            check_reply(reply, category="toy")

        assert "third_person" in error.value.detail

    @pytest.mark.parametrize(
        "reply",
        ["房间内有玩偶和一些纸箱。", "地上有一个玩偶。", "墙上挂着好几个玩偶。", "旁边有玩偶和设备。"],
    )
    def test_describing_the_scene_is_refused(self, reply):
        """The closed fields cannot catch this.

        They say what was confirmed, not what the sentence turned out to be
        about — each of these names the confirmed category and is still a
        remark about the room.
        """
        with pytest.raises(SemanticsError) as error:
            check_reply(reply, category="toy")

        assert "scene_not_visitor" in error.value.detail


class TestNoVocabularyGate:
    def test_an_unlisted_ordinary_word_is_not_a_refusal(self):
        """The regression guard: expression is open.

        "精致" and "毛茸茸" appeared in no whitelist and describe the subject
        perfectly well. If a vocabulary gate returns, these start failing.
        """
        assert accepts("胸前那枚徽章真精致。", "badge")
        assert accepts("这个毛茸茸的玩偶真好玩。", "toy")
