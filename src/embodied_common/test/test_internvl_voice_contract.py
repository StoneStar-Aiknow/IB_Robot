"""The internvl_voice contract: one section, two builders, two nodes.

A wrong value here must never silence the mirror or bring back synthesis, so
the rule under test throughout is: a bad item warns and keeps its default,
everything else goes on.
"""

import json
import subprocess
import sys

import pytest
import yaml

from embodied_common import internvl_voice_contract as contract


def _section(**changes):
    """The §2.1 block as written in the deployment config, with ``changes`` on top."""
    section = {
        "enabled": True,
        "clips_table": "$(find robot_config)/config/mirror_voice/x2_clips_v1.yaml",
        "clip_dir": "assets/audio/x2_mirror_clips_v1",
        "asr": "mute",
        "asr_resume_delay_sec": 0.8,
        "asr_control_topic": "/voice_control",
        "standby": {"first_after_sec": [1.0, 3.0], "gap_sec": [2.0, 5.0]},
        "look": {
            "enabled": True,
            "demo_root": "/root/vlx_internvl_310p",
            "round_period_sec": 0.0,
            "window_sec": 0.0,
            "quick_gap_sec": 2.0,
            "nice": 10,
            "crop_actor": True,
            "refine_hand": True,
            "verify": True,
            "first_after_sec": [2.0, 4.0],
            "gap_sec": [3.0, 6.0],
            "max_per_round": 5,
            "fresh_sec": 5.0,
            "pool_cooldown_sec": 15.0,
            "fallback_after_sec": 2.5,
            "latest_start_sec": 2.0,
            "dump_dir": "",
        },
    }
    section.update(changes)
    return section


TABLE = {
    "version": 1,
    "wav_suffix": "_v1.wav",
    "clips": {
        "evt_starting_01": "启动中，稍等一下。",
        "evt_ready_01": "我准备好咯！",
        "evt_mirroring_01": "开始模仿咯！",
        "evt_goodbye_01": "下次再见吧！",
        "evt_goodbye_02": "玩得真开心，下次再来哦。",
        "standby_any_01": "你好呀！",
        "hand_heart_01": "哇，收到你的爱心啦！",
        "hand_heart_02": "嘿嘿，被你的爱心击中了。",
        "seq_invite_01": "你看，这一下我学得像吗？",
    },
}


def test_the_module_stays_dependency_free():
    """Both launch builders import it while building parameters; both nodes at startup."""
    script = (
        "import sys; import embodied_common.internvl_voice_contract; "
        "leaked = sorted("
        "  name for name in sys.modules"
        "  if name.split('.')[0] in {'rclpy', 'launch', 'robot_config', 'ibrobot_msgs',"
        "                            'manipulation_execution', 'visual_interaction', 'numpy'}"
        ");"
        "assert not leaked, leaked"
    )

    subprocess.run([sys.executable, "-c", script], check=True)


class TestSwitch:
    def test_a_section_left_out_is_the_baseline(self):
        assert contract.normalize(None) == (None, [])

    def test_enabled_false_is_the_baseline(self):
        assert contract.normalize(_section(enabled=False)) == (None, [])

    def test_the_written_block_is_exactly_the_defaults(self):
        settings, warnings = contract.normalize(_section())

        assert warnings == []
        assert settings == contract.Settings()

    @pytest.mark.parametrize("value", ["ture", 1, "false", None, "yes"])
    def test_any_other_value_warns_and_is_on(self, value):
        settings, warnings = contract.normalize(_section(enabled=value))

        assert settings == contract.Settings()
        assert len(warnings) == 1
        assert warnings[0].startswith("internvl_voice.enabled:")

    def test_a_block_without_enabled_warns_and_is_on(self):
        """The block present and not ``enabled: false`` is prerecorded mode (§2.2)."""
        section = _section()
        del section["enabled"]

        settings, warnings = contract.normalize(section)

        assert settings == contract.Settings()
        assert warnings == ["internvl_voice.enabled: missing; read as true"]

    def test_a_section_that_is_not_a_mapping_is_on_with_defaults(self):
        settings, warnings = contract.normalize(["enabled"])

        assert settings == contract.Settings()
        assert warnings and warnings[0].startswith("internvl_voice:")

    def test_false_in_place_of_the_section_is_off(self):
        settings, warnings = contract.normalize(False)

        assert settings is None
        assert warnings


class TestItems:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("clips_table", ""),
            ("clips_table", 3),
            ("clip_dir", None),
            ("asr", "silence"),
            ("asr_resume_delay_sec", -1),
            ("asr_resume_delay_sec", "0.8"),
            ("asr_resume_delay_sec", True),
            ("asr_control_topic", "voice_control"),
        ],
    )
    def test_a_bad_top_item_warns_and_keeps_its_default(self, key, value):
        settings, warnings = contract.normalize(_section(**{key: value}))

        assert settings == contract.Settings()
        assert len(warnings) == 1
        assert warnings[0].startswith(f"internvl_voice.{key}:")

    def test_the_other_items_are_kept(self):
        settings, warnings = contract.normalize(_section(asr="silence", clip_dir="/data/clips", asr_resume_delay_sec=1))

        assert len(warnings) == 1
        assert settings.asr == "mute"
        assert settings.clip_dir == "/data/clips"
        assert settings.asr_resume_delay_sec == 1.0
        assert isinstance(settings.asr_resume_delay_sec, float)

    def test_an_unknown_key_warns_and_is_ignored(self):
        settings, warnings = contract.normalize(_section(asr_mode="mute"))

        assert settings == contract.Settings()
        assert len(warnings) == 1
        assert "asr_mode" in warnings[0]

    def test_a_missing_item_takes_its_default_quietly(self):
        settings, warnings = contract.normalize({"enabled": True})

        assert (settings, warnings) == (contract.Settings(), [])

    @pytest.mark.parametrize("value", [[3.0, 1.0], [1.0], "1-3", [1.0, "3"], [-1.0, 3.0], [1.0, 100.0]])
    def test_a_bad_interval_keeps_its_default(self, value):
        settings, warnings = contract.normalize(_section(standby={"first_after_sec": value, "gap_sec": [2, 4]}))

        assert settings.standby == contract.StandbySettings(first_after_sec=(1.0, 3.0), gap_sec=(2.0, 4.0))
        assert len(warnings) == 1
        assert warnings[0].startswith("internvl_voice.standby.first_after_sec:")

    def test_intervals_are_float_pairs(self):
        settings, _ = contract.normalize(_section(standby={"first_after_sec": [1, 2]}))

        assert settings.standby.first_after_sec == (1.0, 2.0)
        assert all(isinstance(item, float) for item in settings.standby.first_after_sec)

    def test_a_standby_that_is_not_a_mapping_keeps_the_defaults(self):
        settings, warnings = contract.normalize(_section(standby=[1, 3]))

        assert settings.standby == contract.StandbySettings()
        assert len(warnings) == 1


class TestLook:
    def test_look_off_is_none_without_a_warning(self):
        settings, warnings = contract.normalize(_section(look={"enabled": False, "nice": "x"}))

        assert settings.look is None
        assert warnings == []

    @pytest.mark.parametrize("value", [None, "on", [1], True])
    def test_a_broken_look_is_off_and_playback_goes_on(self, value):
        settings, warnings = contract.normalize(_section(look=value))

        assert settings.look is None
        assert settings.standby == contract.StandbySettings()
        assert len(warnings) == 1
        assert warnings[0].startswith("internvl_voice.look:")

    def test_a_look_left_out_is_the_default_look(self):
        section = _section()
        del section["look"]

        assert contract.normalize(section) == (contract.Settings(), [])

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("round_period_sec", -1),
            ("window_sec", -0.5),
            ("nice", -5),
            ("nice", 2.5),
            ("crop_actor", "yes"),
            ("max_per_round", 0),
            ("gap_sec", [6.0, 3.0]),
            ("dump_dir", None),
            ("demo_root", ""),
        ],
    )
    def test_a_bad_look_item_warns_and_keeps_its_default(self, key, value):
        look = dict(_section()["look"], **{key: value})

        settings, warnings = contract.normalize(_section(look=look))

        assert settings.look == contract.LookSettings()
        assert len(warnings) == 1
        assert warnings[0].startswith(f"internvl_voice.look.{key}:")

    @pytest.mark.parametrize(("key", "value"), [("window_sec", 4), ("round_period_sec", 2.5)])
    def test_a_window_or_a_round_period_may_be_set(self, key, value):
        """Both default to 0 (look all the time, back to back); a positive value slows the look."""
        look = dict(_section()["look"], **{key: value})

        settings, warnings = contract.normalize(_section(look=look))

        assert warnings == []
        assert getattr(settings.look, key) == float(value)

    def test_look_enabled_typo_warns_and_is_on(self):
        look = dict(_section()["look"], enabled="ture")

        settings, warnings = contract.normalize(_section(look=look))

        assert settings.look == contract.LookSettings()
        assert len(warnings) == 1

    def test_an_unknown_look_key_warns(self):
        look = dict(_section()["look"], period=1.0)

        settings, warnings = contract.normalize(_section(look=look))

        assert settings.look == contract.LookSettings()
        assert len(warnings) == 1
        assert "period" in warnings[0]


class TestResolve:
    def test_an_absolute_path_is_kept(self):
        assert contract.resolve("/root/vlx_internvl_310p", "/ws") == "/root/vlx_internvl_310p"

    def test_a_relative_path_is_under_the_workspace(self):
        assert contract.resolve("assets/audio/x", "/root/tree") == "/root/tree/assets/audio/x"

    def test_without_a_workspace_a_relative_path_stays_relative(self):
        assert contract.resolve("assets/audio/x", "") == "assets/audio/x"


class TestClipTable:
    def test_a_good_table_parses_without_warnings(self):
        table, warnings = contract.parse_clip_table(TABLE)

        assert warnings == []
        assert table.version == 1
        assert table.wav_suffix == "_v1.wav"
        assert list(table.clips) == list(TABLE["clips"])

    def test_pools_come_from_the_ids_in_table_order(self):
        table, _ = contract.parse_clip_table(TABLE)

        assert table.pool_of("evt_ready_01") == "evt.ready"
        assert table.pool_of("standby_any_01") == "standby.any"
        assert list(table.pools()) == [
            "evt.starting",
            "evt.ready",
            "evt.mirroring",
            "evt.goodbye",
            "standby.any",
            "hand.heart",
            "seq.invite",
        ]
        assert table.pools()["evt.goodbye"] == ("evt_goodbye_01", "evt_goodbye_02")

    def test_the_wav_name(self):
        table, _ = contract.parse_clip_table(TABLE)

        assert table.wav_name("hand_heart_01") == "hand_heart_01_v1.wav"

    @pytest.mark.parametrize(
        "clip_id", ["evt_ready_1", "Evt_ready_01", "evt_ready", "evt_ready_01_x", "evt-ready-01", 7]
    )
    def test_a_bad_id_is_dropped(self, clip_id):
        table, warnings = contract.parse_clip_table({**TABLE, "clips": {**TABLE["clips"], clip_id: "好的呀。"}})

        assert clip_id not in table.clips
        assert len(table.clips) == len(TABLE["clips"])
        assert len(warnings) == 1
        assert "dropped" in warnings[0]

    @pytest.mark.parametrize("text", ["", "   ", None, 3, "比个V吧", "第2次了", "OK"])
    def test_a_bad_text_is_dropped(self, text):
        table, warnings = contract.parse_clip_table({**TABLE, "clips": {**TABLE["clips"], "other_any_01": text}})

        assert "other_any_01" not in table.clips
        assert len(warnings) == 1

    def test_a_repeated_text_is_dropped(self):
        clips = {**TABLE["clips"], "other_any_01": "你好呀！"}

        table, warnings = contract.parse_clip_table({**TABLE, "clips": clips})

        assert "other_any_01" not in table.clips
        assert table.clips["standby_any_01"] == "你好呀！"
        assert len(warnings) == 1
        assert "standby_any_01" in warnings[0]

    def test_a_missing_required_pool_warns(self):
        clips = {key: text for key, text in TABLE["clips"].items() if not key.startswith("standby")}

        table, warnings = contract.parse_clip_table({**TABLE, "clips": clips})

        assert contract.missing_pools(table) == ["standby.any"]
        assert warnings == ["clips_table: pool standby.any has no clip"]

    def test_a_bad_head_keeps_its_default(self):
        table, warnings = contract.parse_clip_table({**TABLE, "version": "one", "wav_suffix": "../x.wav", "extra": 1})

        assert (table.version, table.wav_suffix) == (1, "_v1.wav")
        assert len(warnings) == 3

    @pytest.mark.parametrize("mapping", [None, [], "clips"])
    def test_a_table_that_is_not_a_mapping_is_empty(self, mapping):
        table, warnings = contract.parse_clip_table(mapping)

        assert dict(table.clips) == {}
        assert warnings


@pytest.fixture()
def share(tmp_path):
    """A fake robot_config share dir holding the installed table."""
    root = tmp_path / "share" / "robot_config"
    (root / "config" / "mirror_voice").mkdir(parents=True)
    path = root / "config" / "mirror_voice" / "x2_clips_v1.yaml"
    path.write_text(yaml.safe_dump(TABLE, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return root


def _find(share):
    def find(package):
        assert package == "robot_config"
        return str(share)

    return find


class TestBuildBundle:
    def test_off_is_an_empty_bundle(self, share):
        bundle = contract.build_bundle(_section(enabled=False), "/ws", "/nowhere.yaml", _find(share))

        assert bundle == contract.Bundle(None, None, (), ())

    def test_the_written_block(self, share):
        default = str(share / "config" / "mirror_voice" / "x2_clips_v1.yaml")

        bundle = contract.build_bundle(_section(), "/root/tree", default, _find(share))

        assert bundle.warnings == ()
        assert list(bundle.table.clips) == list(TABLE["clips"])
        assert bundle.clip_dirs == ("/root/tree/assets/audio/x2_mirror_clips_v1",)
        assert bundle.settings.clips_table == default
        assert bundle.settings.look.demo_root == "/root/vlx_internvl_310p"

    def test_the_configured_dir_comes_before_the_default(self, share, tmp_path):
        default = str(share / "config" / "mirror_voice" / "x2_clips_v1.yaml")

        bundle = contract.build_bundle(
            _section(clip_dir="/data/clips", look={"demo_root": "models/vlx", "enabled": True}),
            "/root/tree",
            default,
            _find(share),
        )

        assert bundle.clip_dirs == ("/data/clips", "/root/tree/assets/audio/x2_mirror_clips_v1")
        assert bundle.settings.look.demo_root == "/root/tree/models/vlx"

    @pytest.mark.parametrize("content", [None, "clips: [\n", "clips: {evt_ready_01: 好}\n"])
    def test_an_unusable_table_falls_back_to_the_installed_one(self, share, tmp_path, content):
        configured = tmp_path / "mine.yaml"
        if content is not None:
            configured.write_text(content, encoding="utf-8")
        default = str(share / "config" / "mirror_voice" / "x2_clips_v1.yaml")

        bundle = contract.build_bundle(_section(clips_table=str(configured)), "/ws", default, _find(share))

        assert list(bundle.table.clips) == list(TABLE["clips"])
        assert bundle.settings.clips_table == default
        assert any("using the installed table" in warning for warning in bundle.warnings)

    def test_a_find_that_fails_falls_back_to_the_installed_one(self, share):
        default = str(share / "config" / "mirror_voice" / "x2_clips_v1.yaml")

        def find(package):
            raise LookupError(package)

        bundle = contract.build_bundle(_section(), "/ws", default, find)

        assert list(bundle.table.clips) == list(TABLE["clips"])
        assert bundle.warnings

    def test_no_usable_table_at_all_is_an_empty_table_not_off(self, tmp_path):
        bundle = contract.build_bundle(_section(), "/ws", str(tmp_path / "gone.yaml"), lambda package: str(tmp_path))

        assert bundle.settings is not None
        assert dict(bundle.table.clips) == {}
        assert any("nothing to play" in warning for warning in bundle.warnings)


class TestPlaybackJson:
    @pytest.fixture()
    def payload(self):
        settings, _ = contract.normalize(_section())
        table, _ = contract.parse_clip_table(TABLE)
        return contract.playback_json(settings, table, ["/a", "/b"])

    def test_it_round_trips(self, payload):
        data, warnings = contract.parse_playback_json(payload)

        assert warnings == []
        assert data == {
            "version": 1,
            "clip_dirs": ["/a", "/b"],
            "wav_suffix": "_v1.wav",
            "clips": dict(TABLE["clips"]),
            "events": {
                "starting": ["evt_starting_01"],
                "ready": ["evt_ready_01"],
                "mirroring": ["evt_mirroring_01"],
                "goodbye": ["evt_goodbye_01", "evt_goodbye_02"],
            },
            "standby": {"ids": ["standby_any_01"], "first_after_sec": (1.0, 3.0), "gap_sec": (2.0, 5.0)},
            "asr": {"mode": "mute", "resume_delay_sec": 0.8, "topic": "/voice_control"},
        }

    def test_it_is_plain_utf8_json(self, payload):
        assert "启动中" in payload

    def test_a_bad_field_keeps_its_default(self, payload):
        data = json.loads(payload)
        data["asr"]["mode"] = "loud"
        data["standby"]["gap_sec"] = [5, 2]
        data["events"]["ready"].append("evt_ready_09")

        parsed, warnings = contract.parse_playback_json(json.dumps(data))

        assert parsed["asr"]["mode"] == "mute"
        assert parsed["standby"]["gap_sec"] == (2.0, 5.0)
        assert parsed["events"]["ready"] == ["evt_ready_01"]
        assert len(warnings) == 3

    @pytest.mark.parametrize("text", ["", "{", "[]", None])
    def test_garbage_is_every_default(self, text):
        data, warnings = contract.parse_playback_json(text)

        assert warnings
        assert data["clips"] == {}
        assert data["events"] == {event: [] for event in contract.EVENTS}
        assert data["clip_dirs"] == ["assets/audio/x2_mirror_clips_v1"]
        assert data["asr"]["mode"] == "mute"


class TestLookJson:
    @pytest.fixture()
    def settings_and_table(self):
        settings, _ = contract.normalize(_section())
        table, _ = contract.parse_clip_table(TABLE)
        return settings, table

    def test_it_carries_only_look_pools(self, settings_and_table):
        data = json.loads(contract.look_json(*settings_and_table))

        assert data["pools"] == {
            "hand.heart": [["hand_heart_01", "哇，收到你的爱心啦！"], ["hand_heart_02", "嘿嘿，被你的爱心击中了。"]],
            "seq.invite": [["seq_invite_01", "你看，这一下我学得像吗？"]],
        }

    def test_it_carries_every_look_key(self, settings_and_table):
        data = json.loads(contract.look_json(*settings_and_table))

        assert set(data) == {*contract.LOOK_KEYS, "pools"}
        assert set(contract.LOOK_KEYS) == set(_section()["look"])

    def test_it_round_trips(self, settings_and_table):
        data, warnings = contract.parse_look_json(contract.look_json(*settings_and_table))

        assert warnings == []
        expected = {key: getattr(contract.LookSettings(), key) for key in contract.LOOK_KEYS}
        assert {key: data[key] for key in contract.LOOK_KEYS} == expected
        assert data["pools"]["seq.invite"] == (("seq_invite_01", "你看，这一下我学得像吗？"),)

    def test_a_look_that_is_off_has_no_json(self, settings_and_table):
        settings, table = settings_and_table
        settings, _ = contract.normalize(_section(look={"enabled": False}))

        with pytest.raises(ValueError, match="look is off"):
            contract.look_json(settings, table)

    def test_bad_fields_and_foreign_pools_are_dropped(self, settings_and_table):
        data = json.loads(contract.look_json(*settings_and_table))
        data["nice"] = "low"
        data["pools"]["evt.ready"] = [["evt_ready_01", "我准备好咯！"]]
        data["pools"]["seq.invite"].append(["hand_heart_01", "错池"])

        parsed, warnings = contract.parse_look_json(json.dumps(data))

        assert parsed["nice"] == 10
        assert "evt.ready" not in parsed["pools"]
        assert parsed["pools"]["seq.invite"] == (("seq_invite_01", "你看，这一下我学得像吗？"),)
        assert len(warnings) == 3

    def test_garbage_is_every_default(self):
        data, warnings = contract.parse_look_json("not json")

        assert warnings
        assert data["pools"] == {}
        assert data["round_period_sec"] == 0.0
