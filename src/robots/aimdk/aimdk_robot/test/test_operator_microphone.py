import pytest
from aimdk_robot.operator_microphone import select_microphone


class Backend:
    def __init__(self, readings, error=None):
        self.readings = iter(readings)
        self.writes = []
        self.error = error

    def get_source(self):
        return next(self.readings)

    def set_source(self, source):
        self.writes.append(source)
        if self.error:
            raise self.error


def test_external_selection_waits_for_readback():
    backend = Backend([0, 0, 1])
    result = select_microphone(backend, "external", pause=lambda _: None)
    assert result == {"before": 0, "selected": 1, "source": "external", "changed": True}
    assert backend.writes == [1]


def test_selected_source_is_not_written_again():
    backend = Backend([1, 1])
    assert not select_microphone(backend, "external")["changed"]
    assert backend.writes == []


def test_builtin_selection_uses_source_zero():
    backend = Backend([1, 0])
    assert select_microphone(backend, "builtin")["selected"] == 0
    assert backend.writes == [0]


def test_failed_readback_does_not_retry_switch():
    backend = Backend([0] * 7)
    with pytest.raises(RuntimeError, match="no write retry"):
        select_microphone(backend, "external", pause=lambda _: None)
    assert backend.writes == [1]


def test_unknown_write_result_is_not_retried():
    backend = Backend([0], RuntimeError("result unknown"))
    with pytest.raises(RuntimeError, match="result unknown"):
        select_microphone(backend, "external")
    assert backend.writes == [1]
