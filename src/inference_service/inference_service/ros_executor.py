"""The Humble executor's private wait-set invalidation boundary."""


def release_cached_wait_set(executor):
    """Call only on the spin owner between spin_once calls, never in a callback.

    Humble has no public invalidation API. Keep the executor itself: replacing
    it loses queued Future callbacks. Callers still own worker checks and entity
    retirement; releasing this iterator alone does not prove worker quiescence.
    """
    iterator = executor._cb_iter
    executor._cb_iter = None
    if iterator is not None:
        iterator.close()
