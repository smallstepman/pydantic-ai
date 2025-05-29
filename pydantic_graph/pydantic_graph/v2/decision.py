from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Never

from typing_extensions import Literal, get_args, get_origin

from pydantic_graph.v2.id_types import NodeId
from pydantic_graph.v2.join import Join
from pydantic_graph.v2.node import EndNode
from pydantic_graph.v2.step import Step
from pydantic_graph.v2.transform import TransformFunction
from pydantic_graph.v2.util import TypeExpression

if TYPE_CHECKING:
    from pydantic_graph.v2.node_types import AnyDestinationNode


@dataclass
class Decision[StateT, DepsT, SourceT, EndT]:
    id: NodeId
    branches: list[DecisionBranch[StateT, DepsT, Any, Any]]
    # TODO: Add a field for the label for the input edge
    note: str | None

    def branch[S, E, S2, E2](
        self: Decision[StateT, DepsT, S, E], branch: DecisionBranch[StateT, DepsT, S2, E2]
    ) -> Decision[StateT, DepsT, S | S2, E | E2]:
        return Decision(id=self.id, branches=self.branches + [branch], note=self.note)

    def _force_source_invariant(self, source: SourceT) -> SourceT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')

    def _force_end_covariant(self) -> EndT:
        raise RuntimeError('This method should never be called, it is just defined for typing purposes.')


@dataclass
class DecisionBranch[StateT, DepsT, SourceT, EndT]:
    source: type[SourceT]
    # TODO: Support broadcast forks by allowing `route_to` to be a sequence of nodes(?)
    route_to: AnyDestinationNode
    # TODO: Rename `matches` to `test_match` or similar
    matches: Callable[[Any], bool] | None = None
    # TODO: If we change `transforms` to a single callable, we can make SourceT the type of the inputs
    transforms: tuple[TransformFunction[StateT, DepsT, Any, Any, Any], ...] = ()
    user_label: str | None = None
    # TODO: the branch needs a node ID to use as the ID of the spread node
    spread: bool = False
    post_spread_transform: TransformFunction[StateT, DepsT, Any, Any, Any] | None = None
    post_spread_user_label: str | None = None

    @property
    def label(self) -> str | None:
        if self.user_label:
            return self.user_label

        source = self.source
        if get_origin(self.source) is TypeExpression:
            source = get_args(self.source)[0]

        if get_origin(source) is Literal:
            return ', '.join(repr(arg) for arg in get_args(source))

        return getattr(source, '__name__', str(self.source))

    @property
    def post_spread_label(self) -> str | None:
        return self.post_spread_user_label


@dataclass
class DecisionBranchBuilder[StateT, DepsT, SourceT, EdgeInputT, EdgeOutputT]:
    source: type[SourceT]
    matches: Callable[[Any], bool] | None = None
    transforms: tuple[TransformFunction[StateT, DepsT, EdgeInputT, Any, Any], ...] = ()
    user_label: str | None = None

    def transform[T](
        self,
        call: TransformFunction[StateT, DepsT, EdgeInputT, EdgeOutputT, T],
    ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, EdgeInputT, T]:
        new_transforms = self.transforms + (call,)
        return DecisionBranchBuilder(self.source, self.matches, new_transforms)

    def route_to(  # analogous to GraphBuilder.edge
        self,
        # TODO: Can we support broadcast forks somehow?
        node: Step[StateT, DepsT, EdgeOutputT, Any]
        | Join[StateT, DepsT, EdgeOutputT, Any]
        | Decision[StateT, DepsT, EdgeOutputT, Any],
    ) -> DecisionBranch[StateT, DepsT, SourceT, Never]:
        return DecisionBranch[StateT, DepsT, SourceT, Never](
            source=self.source,
            route_to=node,
            matches=self.matches,
            transforms=self.transforms,
            user_label=self.user_label,
        )

    def spread_to[T](  # analogous to GraphBuilder.spreading_edge
        self: DecisionBranchBuilder[StateT, DepsT, SourceT, EdgeInputT, Sequence[T]],
        node: Step[StateT, DepsT, EdgeOutputT, Any]
        | Join[StateT, DepsT, EdgeOutputT, Any]
        | Decision[StateT, DepsT, EdgeOutputT, Any],
        post_spread_transform: TransformFunction[StateT, DepsT, Sequence[T], Any, Any] | None = None,
        post_spread_user_label: str | None = None,
    ) -> DecisionBranch[StateT, DepsT, SourceT, Never]:
        return DecisionBranch[StateT, DepsT, SourceT, Never](
            source=self.source,
            route_to=node,
            matches=self.matches,
            transforms=self.transforms,
            user_label=self.user_label,
            spread=True,
            post_spread_transform=post_spread_transform,
            post_spread_user_label=post_spread_user_label,
        )

    def end(
        self,
    ) -> DecisionBranch[StateT, DepsT, SourceT, EdgeOutputT]:
        return DecisionBranch(
            source=self.source,
            route_to=EndNode.end,
            matches=self.matches,
            transforms=self.transforms,
            user_label=self.user_label,
        )
