from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic_graph.v2.id_types import ForkId, JoinId


# TODO: Merge this with StepContext?
class ReducerContext[StateT, DepsT, InputT]:
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    def __init__(self, state: StateT, deps: DepsT, inputs: InputT):
        self._state = state
        self._deps = deps
        self._inputs = inputs

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def deps(self) -> DepsT:
        return self._deps

    @property
    def inputs(self) -> InputT:
        return self._inputs

    def cancel_other_requests(self) -> None:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}(state={self.state}, deps={self.deps}, inputs={self.inputs})'


class Reducer[StateT, DepsT, InputT, OutputT]:
    def __init__(self, state: StateT, deps: DepsT, inputs: InputT):
        self._state = state
        self._deps = deps

    def reduce(self, ctx: ReducerContext[StateT, DepsT, InputT]) -> None:
        raise NotImplementedError

    def finalize(self, ctx: ReducerContext[StateT, DepsT, None]) -> OutputT:
        raise NotImplementedError


class NullReducer(Reducer[Any, Any, Any, Any]):
    def reduce(self, ctx: ReducerContext[Any, Any, Any]) -> None:
        # No-op reducer
        pass

    def finalize(self, ctx: ReducerContext[Any, Any, None]) -> Any:
        return None  # or whatever default value is appropriate


# TODO: Make this accept a single context input rather than three inputs
type ReducerFactory[StateT, DepsT, InputT, OutputT] = Callable[
    [StateT, DepsT, InputT], Reducer[StateT, DepsT, InputT, OutputT]
]


def list_reducer[T](item_type: type[T]) -> type[Reducer[object, object, T, list[T]]]:
    # append to list
    raise NotImplementedError


def dict_reducer[T: dict[Any, Any]](
    dict_type: type[T],
) -> type[Reducer[object, object, T, T]]:
    # update dict
    raise NotImplementedError


@dataclass
class Join[StateT, DepsT, InputT, OutputT]:
    id: JoinId

    reducer_factory: ReducerFactory[StateT, DepsT, InputT, OutputT]

    # TODO: Need to implement a version of ParentForkFinder that validates the specified NodeId is valid
    joins: ForkId | None = None  # the NodeID of the node to use as the dominating fork
