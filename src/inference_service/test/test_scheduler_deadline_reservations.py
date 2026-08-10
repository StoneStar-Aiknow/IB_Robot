import threading
import time

from inference_service.scheduler.deadline_reservations import DeadlineReservationTable


def test_same_resource_reservations_are_serialized() -> None:
    table = DeadlineReservationTable()
    first = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=300,
        estimate_ns=80,
    )
    second = table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=110,
        deadline_ns=300,
        estimate_ns=90,
    )

    assert first is not None
    assert second is not None
    assert (first.estimated_start_ns, first.estimated_finish_ns) == (100, 180)
    assert (second.estimated_start_ns, second.estimated_finish_ns) == (180, 270)


def test_same_resource_dispatch_waits_for_predecessor_release() -> None:
    table = DeadlineReservationTable()
    now_ns = time.monotonic_ns()
    first = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=now_ns,
        deadline_ns=now_ns + 1_000_000_000,
        estimate_ns=20_000_000,
    )
    second = table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=now_ns,
        deadline_ns=now_ns + 1_000_000_000,
        estimate_ns=20_000_000,
    )
    assert first is not None and second is not None
    completed = threading.Event()
    result: list[str] = []

    waiter = threading.Thread(
        target=lambda: (result.append(table.wait_for_turn(second, deadline_ns=now_ns + 1_000_000_000)), completed.set())
    )
    waiter.start()
    assert not completed.wait(0.05)
    table.release(first)
    assert completed.wait(1.0)
    waiter.join(timeout=1.0)
    assert result == ["ready"]
    table.release(second)


def test_dispatch_turn_rechecks_deadline_against_actual_start() -> None:
    table = DeadlineReservationTable()
    first = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=1_000,
        estimate_ns=100,
    )
    second = table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=350,
        estimate_ns=100,
    )
    assert first is not None and second is not None
    table.release(first)

    assert table.wait_for_turn(second, deadline_ns=350, now_ns=lambda: 300) == "deadline_exceeded"


def test_reservation_rejects_when_existing_work_exhausts_deadline() -> None:
    table = DeadlineReservationTable()
    assert table.try_reserve(
        pipeline_id="first",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=300,
        estimate_ns=150,
    )
    assert (
        table.try_reserve(
            pipeline_id="second",
            hardware_resource_id="ascend:0",
            now_ns=110,
            deadline_ns=300,
            estimate_ns=60,
        )
        is None
    )


def test_different_resources_do_not_share_a_limit() -> None:
    table = DeadlineReservationTable()
    first = table.try_reserve(
        pipeline_id="first",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=200,
        estimate_ns=100,
    )
    second = table.try_reserve(
        pipeline_id="second",
        hardware_resource_id="ascend:1",
        now_ns=100,
        deadline_ns=200,
        estimate_ns=100,
    )

    assert first is not None
    assert second is not None
    assert second.estimated_start_ns == 100


def test_not_started_and_completed_work_release_reservations() -> None:
    table = DeadlineReservationTable()
    reservation = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=200,
        estimate_ns=100,
    )
    assert reservation is not None

    table.release(reservation)

    replacement = table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=110,
        deadline_ns=210,
        estimate_ns=100,
    )
    assert replacement is not None
    assert replacement.estimated_start_ns == 110


def test_unknown_quarantines_resource_until_pipeline_reboot() -> None:
    table = DeadlineReservationTable()
    reservation = table.try_reserve(
        pipeline_id="primary",
        hardware_resource_id="ascend:0",
        now_ns=100,
        deadline_ns=200,
        estimate_ns=100,
    )
    assert reservation is not None
    table.mark_unknown(reservation)

    assert (
        table.try_reserve(
            pipeline_id="fallback",
            hardware_resource_id="ascend:0",
            now_ns=300,
            deadline_ns=500,
            estimate_ns=100,
        )
        is None
    )

    table.reconcile_pipeline("primary")
    assert table.try_reserve(
        pipeline_id="fallback",
        hardware_resource_id="ascend:0",
        now_ns=300,
        deadline_ns=500,
        estimate_ns=100,
    )
