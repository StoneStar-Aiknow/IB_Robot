"""Errors independent of a terminal's output or exit-code policy."""


class ClientError(RuntimeError):
    def __init__(self, code: str, message: str, *, reference=None, observation_unknown=False):
        super().__init__(message)
        self.code = code
        self.reference = reference
        self.observation_unknown = observation_unknown
