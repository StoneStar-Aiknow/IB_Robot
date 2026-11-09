"""Single-flight candidate preparation without executor interpreter-exit joins."""

import threading
from concurrent.futures import Future


class PolicyLoader:
    """Run read-only preparation only; callers poll results and own activation.

    A blocked load may outlive close as a daemon. It must not mutate live state or
    call back into the node. No attempt is made to terminate native code, which
    can still prevent shutdown if it holds the GIL.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._closed = False

    @property
    def busy(self):
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def submit(self, function, *args):
        with self._lock:
            if self._closed:
                raise RuntimeError("Policy loader is closed")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Policy loader is busy")
            future = Future()
            thread = threading.Thread(
                target=self._run,
                args=(future, function, args),
                name="simulation-policy-load",
                daemon=True,
            )
            thread.start()
            self._thread = thread
            return future

    @staticmethod
    def _run(future, function, args):
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = function(*args)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    def close(self, timeout=0.5):
        """Reject further jobs, wait at most timeout seconds, and report completion."""
        with self._lock:
            self._closed = True
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()
