"""Execution errors shared by generated and fixed-library poses."""


class ComposeError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
