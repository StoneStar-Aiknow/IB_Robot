"""Measurement provenance: what a report records, and what it refuses to."""

import json

import pytest

from visual_interaction.phase0.report import (
    SCHEMA,
    build_report,
    digests,
    environment,
    sha256_file,
    source_state,
    write_report,
)


class TestDigests:
    def test_a_digest_identifies_the_contents(self, tmp_path):
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        first.write_bytes(b"same")
        second.write_bytes(b"same")

        assert sha256_file(first) == sha256_file(second)

    def test_a_changed_byte_changes_the_digest(self, tmp_path):
        path = tmp_path / "a.bin"
        path.write_bytes(b"one")
        before = sha256_file(path)
        path.write_bytes(b"two")

        assert sha256_file(path) != before

    def test_assets_are_keyed_by_label_not_by_path(self, tmp_path):
        """§13 keeps deployment paths out of the repository; a label is enough."""
        path = tmp_path / "secret-location.om"
        path.write_bytes(b"model")
        result = digests({"om:vision": path})

        assert list(result) == ["om:vision"]
        assert str(path) not in json.dumps(result)


class TestSourceState:
    def test_a_clean_checkout_is_reproducible(self, tmp_path):
        import subprocess

        repo = tmp_path / "repo"
        repo.mkdir()
        for args in (
            ["init", "-q"],
            ["config", "user.email", "t@example.com"],
            ["config", "user.name", "T"],
        ):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
        (repo / "f.txt").write_text("hello")
        subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "c"], check=True, capture_output=True)

        state = source_state(repo)
        assert state["reproducible"] is True
        assert len(state["commit"]) == 40

        (repo / "f.txt").write_text("changed")
        dirty = source_state(repo)
        assert dirty["reproducible"] is False
        assert dirty["dirty_count"] == 1

    def test_outside_a_checkout_it_is_not_reproducible_rather_than_an_error(self, tmp_path):
        """An exploratory run still produces a report; it just cannot be cited."""
        state = source_state(tmp_path)

        assert state["commit"] is None
        assert state["reproducible"] is False


class TestEnvironment:
    def test_it_records_the_interpreter_and_platform(self):
        recorded = environment()

        assert recorded["python"]
        assert recorded["machine"]

    def test_it_does_not_record_a_host_name(self):
        """§13 keeps private deployment details out of the repository."""
        import socket

        assert socket.gethostname() not in json.dumps(environment())


class TestBuildReport:
    def report(self, **overrides):
        values = {
            "tool": "token_budget",
            "conditions": {"label": "idle"},
            "results": {"all_fit": True},
            "repo": ".",
        }
        values.update(overrides)
        return build_report(**values)

    def test_it_carries_the_schema_tool_and_provenance(self):
        report = self.report()

        assert report["schema"] == SCHEMA
        assert report["tool"] == "token_budget"
        assert "commit" in report["source"]
        assert report["conditions"] == {"label": "idle"}

    def test_notes_survive_so_a_caveat_travels_with_the_numbers(self):
        report = self.report(notes=["synthesize only; nothing was played"])

        assert report["notes"] == ["synthesize only; nothing was played"]

    def test_it_is_json_serialisable(self):
        """A report that cannot be written is not a measurement."""
        json.dumps(self.report())


class TestWriteReport:
    def test_it_writes_readable_json(self, tmp_path):
        target = write_report(tmp_path / "m.json", {"tool": "t"})

        assert json.loads(target.read_text(encoding="utf-8")) == {"tool": "t"}

    def test_it_creates_the_parent_directory(self, tmp_path):
        assert write_report(tmp_path / "nested" / "m.json", {"tool": "t"}).exists()

    def test_it_refuses_to_overwrite(self, tmp_path):
        """A measurement is written once; editing one destroys the evidence."""
        target = tmp_path / "m.json"
        write_report(target, {"tool": "t"})

        with pytest.raises(FileExistsError, match="written once"):
            write_report(target, {"tool": "t"})
