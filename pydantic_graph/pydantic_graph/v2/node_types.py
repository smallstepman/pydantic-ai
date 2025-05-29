from __future__ import annotations

from typing import Any

from typing_extensions import TypeGuard

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.join import Join
from pydantic_graph.v2.node import EndNode, Spread, StartNode
from pydantic_graph.v2.step import Step

type AnyMiddleNode = Step[Any, Any, Any, Any] | Join[Any, Any, Any, Any] | Spread
type AnySourceNode = AnyMiddleNode | StartNode
type AnyDestinationNode = AnyMiddleNode | EndNode | Decision[Any, Any, Any, Any]
type AnyNode = AnySourceNode | AnyDestinationNode
# TODO: Add a constraint that there is at most one edge or fork between any two nodes.
#   I _think_ any reasonable graph design should be able to work around that restriction, and then the
#   the only thing that need (unique) IDs for state persistence are steps and joins.
#   Note that to serialize graph run state, you need to serialize the walk states (position + fork stack) and reducers,
#   and fork stacks need to reference the source fork ID. But in a world where there is precisely one edge/fork between
#   nodes, you should be able to build a fork ID from the parent node id (for broadcasts) or parent + child node IDs (for spreads).
#   Note that this assumes that decisions and spreads do not use "heavy" logic and do not need to be involved with persistence.
#   Note that it may also be possible/desirable to drop the need for an ID on joins, but we'll at least need to serialize reducers.


def is_source(node: AnyNode) -> TypeGuard[AnySourceNode]:
    return isinstance(node, (StartNode, Step, Join))


def is_destination(node: AnyNode) -> TypeGuard[AnyDestinationNode]:
    return isinstance(node, (EndNode, Step, Join, Decision))
