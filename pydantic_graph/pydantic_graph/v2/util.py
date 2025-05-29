import inspect
import uuid
from typing import Any


class TypeExpression[T]:
    pass


def get_callable_name(callable_: Any) -> str:
    # TODO: Need to improve this...
    return getattr(callable_, '__name__', str(callable_))


def get_unique_string() -> str:
    # TODO: Need to improve this...
    return str(uuid.uuid4()).replace('-', '')


def infer_name(obj: Any, *, depth: int) -> str | None:
    """Infer the name of `obj` from the call frame.

    Usage should generally look like `infer_name(self, inspect.currentframe())`.
    """
    target_frame = inspect.currentframe()
    if target_frame is None:
        return None
    for _ in range(depth):
        target_frame = target_frame.f_back
        if target_frame is None:
            return None

    for name, item in target_frame.f_locals.items():
        if item is obj:
            return name

    if target_frame.f_locals != target_frame.f_globals:
        # if we couldn't find the agent in locals and globals are a different dict, try globals
        for name, item in target_frame.f_globals.items():
            if item is obj:
                return name

    return None
