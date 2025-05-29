from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from pydantic_graph.v2.id_types import NodeId


# TODO: Should StepContext be passed to joins/forks/decisions? Like, unified with ReducerContext etc.?
# TODO: Make InputT default to object so it can be dropped when not relevant
# TODO: Do we really need both state and deps???
class StepContext[StateT, DepsT, InputT]:
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

    def __repr__(self):
        return f'{self.__class__.__name__}(state={self.state}, deps={self.deps}, inputs={self.inputs})'


class StepCallProtocol[StateT, DepsT, InputT, OutputT](Protocol):
    """The purpose of this is to make it possible to deserialize step calls similar to how Evaluators work."""

    def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> Awaitable[OutputT]:
        raise NotImplementedError


@dataclass
class Step[StateT, DepsT, InputT, OutputT]:
    id: NodeId
    call: StepCallProtocol[StateT, DepsT, InputT, OutputT]
    user_label: str | None

    @property
    def label(self) -> str | None:
        return self.user_label
