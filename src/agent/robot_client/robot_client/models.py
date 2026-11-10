"""Conversion of transport values to the public JSON-compatible value contract."""

from types import SimpleNamespace


def public_value(value):
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): public_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [public_value(item) for item in value]
    if isinstance(value, SimpleNamespace):
        return public_value(vars(value))
    fields = getattr(value, "get_fields_and_field_types", None)
    if callable(fields):
        return {key: public_value(getattr(value, key)) for key in fields()}
    raise TypeError(f"unsupported public result type: {type(value).__name__}")
