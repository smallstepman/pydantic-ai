from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Never, cast, get_args, get_origin, overload

from anyio import Event, create_memory_object_stream, create_task_group
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from typing_extensions import Literal, Protocol, assert_never

from pydantic_graph.v2.decision import Decision, DecisionBranchBuilder
from pydantic_graph.v2.id_types import ForkId, JoinId, NodeRunId
from pydantic_graph.v2.join import Join, Reducer, ReducerContext, ReducerFactory
from pydantic_graph.v2.mermaid import StateDiagramDirection, generate_code
from pydantic_graph.v2.node import (
    END,
    START,
    EndNode,
    NodeId,
    Spread,
    StartNode,
)
from pydantic_graph.v2.node_types import (
    AnyDestinationNode,
    AnyNode,
    AnySourceNode,
    is_destination,
    is_source,
)
from pydantic_graph.v2.parent_forks import ParentFork, ParentForkFinder
from pydantic_graph.v2.step import Step, StepCallProtocol, StepContext
from pydantic_graph.v2.transform import AnyTransformFunction, TransformContext, TransformFunction
from pydantic_graph.v2.util import TypeExpression, get_callable_name


@dataclass
class Edge:
    source_id: NodeId
    transform: AnyTransformFunction | None
    destination_id: NodeId
    user_label: str | None

    def source(self, nodes: dict[NodeId, AnyNode]) -> AnySourceNode:
        node = nodes.get(self.source_id)
        if node is None:
            raise ValueError(f'Node {self.source_id} not found in graph')
        if not is_source(node):
            raise ValueError(f'Node {self.source_id} is not a source node: {node}')
        return node

    def destination(self, nodes: dict[NodeId, AnyNode]) -> AnyDestinationNode:
        node = nodes.get(self.destination_id)
        if node is None:
            raise ValueError(f'Node {self.destination_id} not found in graph')
        if not is_destination(node):
            raise ValueError(f'Node {self.destination_id} is not a source node: {node}')
        return node

    @property
    def label(self) -> str | None:
        # TODO: Add some default behavior?
        return self.user_label


class HandleBranchProtocol[StateT, DepsT, InputT](Protocol):
    def __call__[SourceT](
        self,
        case: type[SourceT] | type[TypeExpression[SourceT]],
        *,
        matches: Callable[[Any], bool] | None = None,
        label: str | None = None,
    ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, InputT, SourceT]:
        raise NotImplementedError


@dataclass
class Handler[StateT, DepsT, InputT]:
    source: Step[StateT, DepsT, InputT, Any] | Join[StateT, DepsT, InputT, Any]
    handle: HandleBranchProtocol[StateT, DepsT, Any]

    def __call__[SourceT](
        self,
        case: type[SourceT] | type[TypeExpression[SourceT]],
        *,
        matches: Callable[[Any], bool] | None = None,
        label: str | None = None,
    ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, InputT, SourceT]:
        return self.handle(case=case, matches=matches, label=label)


@dataclass
class GraphBuilder[StateT, DepsT, GraphInputT, GraphOutputT]:
    state_type: type[StateT]
    deps_type: type[DepsT]
    input_type: type[GraphInputT]
    output_type: type[TypeExpression[GraphOutputT]] | type[GraphOutputT]

    parallel: bool = True  # if False, allow direct state modification and don't copy state sent to steps, but disallow parallel node execution

    _nodes: dict[NodeId, AnyNode] = field(init=False, default_factory=dict)
    _edges_by_source: dict[NodeId, list[Edge]] = field(init=False, default_factory=lambda: defaultdict(list))
    _decision_index: int = field(init=False, default=1)

    type Source[OutputT] = Step[StateT, DepsT, Any, OutputT] | Join[StateT, DepsT, Any, OutputT]
    type SourceWithInputs[InputT, OutputT] = Step[StateT, DepsT, InputT, OutputT] | Join[StateT, DepsT, InputT, OutputT]
    type Destination[InputT] = (
        Step[StateT, DepsT, InputT, Any]
        | Join[StateT, DepsT, InputT, Any]
        | Decision[StateT, DepsT, InputT, GraphOutputT]
    )

    def __post_init__(self):
        self._nodes[START.id] = START
        self._nodes[END.id] = END

    # Node building:
    @overload
    def step[InputT, OutputT](
        self,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Callable[[StepCallProtocol[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]: ...

    @overload
    def step[InputT, OutputT](
        self,
        call: StepCallProtocol[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Step[StateT, DepsT, InputT, OutputT]: ...

    def step[InputT, OutputT](
        self,
        call: StepCallProtocol[StateT, DepsT, InputT, OutputT] | None = None,
        *,
        node_id: str | None = None,
        label: str | None = None,
    ) -> (
        Step[StateT, DepsT, InputT, OutputT]
        | Callable[[StepCallProtocol[StateT, DepsT, InputT, OutputT]], Step[StateT, DepsT, InputT, OutputT]]
    ):
        if call is None:

            def decorator(
                func: StepCallProtocol[StateT, DepsT, InputT, OutputT],
            ) -> Step[StateT, DepsT, InputT, OutputT]:
                return self.step(call=func, node_id=node_id, label=label)

            return decorator

        if node_id is None:
            node_id = get_callable_name(call)

        if node_id in self._nodes:
            raise ValueError(f'Node ID {node_id!r} is already used in the graph. Please specify a unique ID.')

        node = Step[StateT, DepsT, InputT, OutputT](id=NodeId(node_id), call=call, user_label=label)
        self._nodes[NodeId(node_id)] = node
        return node

    @overload
    def join[InputT, OutputT](
        self,
        *,
        node_id: str | None = None,
    ) -> Callable[[ReducerFactory[StateT, DepsT, InputT, OutputT]], Join[StateT, DepsT, InputT, OutputT]]: ...

    @overload
    def join[InputT, OutputT](
        self,
        reducer_factory: ReducerFactory[StateT, DepsT, InputT, OutputT],
        *,
        node_id: str | None = None,
    ) -> Join[StateT, DepsT, InputT, OutputT]: ...

    def join(
        self,
        reducer_factory: ReducerFactory[StateT, DepsT, Any, Any] | None = None,
        *,
        node_id: str | None = None,
    ) -> (
        Join[StateT, DepsT, Any, Any]
        | Callable[[ReducerFactory[StateT, DepsT, Any, Any]], Join[StateT, DepsT, Any, Any]]
    ):
        if reducer_factory is None:

            def decorator(
                reducer_factory: ReducerFactory[StateT, DepsT, Any, Any],
            ) -> Join[StateT, DepsT, Any, Any]:
                return self.join(reducer_factory=reducer_factory, node_id=node_id)

            return decorator

        if node_id is None:
            # TODO: Ideally we'd be able to infer this from the parent frame variable assignment or similar
            node_id = get_callable_name(reducer_factory)

        if node_id in self._nodes:
            raise ValueError(f'Node ID {node_id!r} is already used in the graph. Please specify a unique ID.')

        node = Join[StateT, DepsT, Any, Any](
            id=JoinId(NodeId(node_id)),
            reducer_factory=reducer_factory,
        )
        self._nodes[NodeId(node_id)] = node
        return node

    def decision(self, *, node_id: str | None = None, note: str | None = None) -> Decision[StateT, DepsT, Never, Never]:
        if node_id is None:
            node_id = self._get_new_decision_id()

        if node_id in self._nodes:
            raise ValueError(f'Node ID {node_id!r} is already used in the graph. Please specify a unique ID.')

        return Decision[StateT, DepsT, Never, Never](id=NodeId(node_id), branches=[], note=note)

    def _get_new_decision_id(self) -> str:
        node_id = f'decision_{self._decision_index}'
        self._decision_index += 1
        while node_id in self._nodes:
            node_id = f'decision_{self._decision_index}'
            self._decision_index += 1
        return node_id

    def _get_new_spread_id(self, from_: str, to: str) -> str:
        prefix = f'spread_from_{from_}_to_{to}'

        node_id = prefix
        index = 2
        while node_id in self._nodes:
            node_id = f'{prefix}_{index}'
            index += 1
        return node_id

    def get_handler[InputT](self, source: SourceWithInputs[InputT, Any]) -> Handler[StateT, DepsT, InputT]:
        return Handler(source=source, handle=self.handle)

    def handle[SourceT](
        self,
        case: type[SourceT] | type[TypeExpression[SourceT]],
        *,
        matches: Callable[[Any], bool] | None = None,
        label: str | None = None,
    ) -> DecisionBranchBuilder[StateT, DepsT, SourceT, Any, SourceT]:
        extracted_case = cast(type[SourceT], get_args(case)[0] if get_origin(case) is TypeExpression else case)
        return DecisionBranchBuilder(extracted_case, matches, transforms=(), user_label=label)

    # Edge building
    # Node "types" to be connected into edges: 'start', 'end', Step, Decision, Join, Fork.
    # You typically don't manually create forks — they are inferred from multiple edges coming out of a single node.
    @overload
    def start_with(
        self,
        destination: Destination[GraphInputT],
        *,
        label: str | None = None,
    ) -> None: ...

    @overload
    def start_with[DestinationInputT](
        self,
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, DepsT, GraphInputT, GraphInputT, DestinationInputT],
        label: str | None = None,
    ) -> None: ...

    def start_with(
        self,
        destination: Destination[Any],
        *,
        transform: TransformFunction[StateT, DepsT, Any, Any, Any] | None = None,
        label: str | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=START,
            transform=transform,
            destination=destination,
            label=label,
        )

    @overload
    def start_with_spread[GraphInputItemT](
        self: GraphBuilder[StateT, DepsT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[GraphInputItemT],
        *,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def start_with_spread[DestinationInputT](
        self,
        node: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, GraphInputT, GraphInputT, Sequence[DestinationInputT]],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def start_with_spread[GraphInputItemT, DestinationInputT](
        self: GraphBuilder[StateT, DepsT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        post_spread_transform: TransformFunction[
            StateT, DepsT, Sequence[GraphInputItemT], GraphInputItemT, DestinationInputT
        ],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def start_with_spread[IntermediateT, DestinationInputT](
        self: GraphBuilder[StateT, DepsT, GraphInputT, GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, GraphInputT, GraphInputT, Sequence[IntermediateT]],
        post_spread_transform: TransformFunction[StateT, DepsT, GraphInputT, IntermediateT, DestinationInputT],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    def start_with_spread(
        self,
        node: Destination[Any],
        *,
        spread_id: str | None = None,
        pre_spread_transform: TransformFunction[StateT, DepsT, Any, Any, Sequence[Any]] | None = None,
        post_spread_transform: TransformFunction[StateT, DepsT, Any, Any, Any] | None = None,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None:
        # TODO: Need to allow specifying the id manually to prevent conflicts if there are multiple spreads between the same two nodes
        if spread_id is None:
            spread_id = self._get_new_spread_id(from_='start', to=node.id)
        if spread_id in self._nodes:
            raise ValueError(f'Spread ID {spread_id!r} is already used in the graph. Please specify a unique ID.')

        spread = Spread(id=ForkId(NodeId(spread_id)))
        self._add_edge_from_nodes(
            source=START,
            transform=pre_spread_transform,
            destination=spread,
            label=pre_spread_label,
        )
        self._add_edge_from_nodes(
            source=spread,
            transform=post_spread_transform,
            destination=node,
            label=post_spread_label,
        )

    @overload
    def add_edge[SourceOutputT](
        self,
        source: Source[SourceOutputT],
        destination: Destination[SourceOutputT],
        *,
        label: str | None = None,
    ) -> None: ...

    @overload
    def add_edge[SourceInputT, SourceOutputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, DepsT, SourceInputT, SourceOutputT, DestinationInputT],
        label: str | None = None,
    ) -> None: ...

    def add_edge(
        self,
        source: Source[Any],
        destination: Destination[Any],
        *,
        transform: AnyTransformFunction | None = None,
        label: str | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=destination,
            label=label,
        )

    @overload
    def add_spreading_edge[SourceInputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, Sequence[DestinationInputT]],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def add_spreading_edge[SourceInputT, SourceOutputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[
            StateT, DepsT, SourceInputT, SourceOutputT, Sequence[DestinationInputT]
        ],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def add_spreading_edge[SourceInputT, SourceOutputItemT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, Sequence[SourceOutputItemT]],
        destination: Destination[DestinationInputT],
        *,
        post_spread_transform: TransformFunction[
            StateT,
            DepsT,
            SourceInputT,
            SourceOutputItemT,
            DestinationInputT,
        ],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    @overload
    def add_spreading_edge[SourceInputT, SourceOutputT, IntermediateT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_spread_transform: TransformFunction[StateT, DepsT, SourceInputT, SourceOutputT, Sequence[IntermediateT]],
        post_spread_transform: TransformFunction[StateT, DepsT, SourceInputT, IntermediateT, DestinationInputT],
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None: ...

    def add_spreading_edge[SourceInputT](
        self,
        source: SourceWithInputs[SourceInputT, Any],
        destination: Destination[Any],
        *,
        spread_id: str | None = None,
        pre_spread_transform: TransformFunction[StateT, DepsT, SourceInputT, Any, Sequence[Any]] | None = None,
        post_spread_transform: TransformFunction[StateT, DepsT, SourceInputT, Any, Any] | None = None,
        pre_spread_label: str | None = None,
        post_spread_label: str | None = None,
    ) -> None:
        # TODO: Need to allow specifying the id manually to prevent conflicts if there are multiple spreads between the same two nodes
        if spread_id is None:
            spread_id = self._get_new_spread_id(from_=source.id, to=destination.id)
        if spread_id in self._nodes:
            raise ValueError(f'Spread ID {spread_id!r} is already used in the graph. Please specify a unique ID.')

        spread = Spread(id=ForkId(NodeId(spread_id)))
        self._add_edge_from_nodes(
            source=source,
            transform=pre_spread_transform,
            destination=spread,
            label=pre_spread_label,
        )
        self._add_edge_from_nodes(
            source=spread,
            transform=post_spread_transform,
            destination=destination,
            label=post_spread_label,
        )

    @overload
    def end_from(
        self,
        source: Source[GraphOutputT],
        *,
        label: str | None = None,
    ) -> None: ...

    @overload
    def end_from[SourceInputT, SourceOutputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        *,
        label: str | None = None,
        transform: TransformFunction[StateT, DepsT, SourceInputT, SourceOutputT, GraphOutputT],
    ) -> None: ...

    def end_from(
        self,
        source: Source[Any],
        *,
        label: str | None = None,
        transform: TransformFunction[StateT, DepsT, Any, Any, GraphOutputT] | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=END,
            label=label,
        )

    def _add_edge_from_nodes(
        self,
        *,
        source: AnySourceNode,
        transform: AnyTransformFunction | None,
        destination: AnyDestinationNode,
        label: str | None = None,
    ) -> None:
        self._insert_node(source)
        self._insert_node(destination)

        edge = Edge(source_id=source.id, transform=transform, destination_id=destination.id, user_label=label)
        self._insert_edge(edge)

    def _insert_node(self, node: AnyNode) -> None:
        existing = self._nodes.get(node.id)
        if existing is None:
            if isinstance(node, Decision):
                # We don't add decisions when first created because the decision-builder API makes new instances rather
                # than mutating. Maybe we can rework it so that only one actual decision is created, in which case
                # we could manage the name more nicely. But maybe not.
                self._nodes[node.id] = node
            else:
                # If we hit the following line, we can remove it and uncomment the following line.
                # But I think it should be unnecessary.
                assert False, f'Nodes should probably be added before edges. {node}'
                # self._nodes[node.id] = node
        elif isinstance(existing, (StartNode, EndNode)):
            pass  # it's not a problem to have non-unique instances of StartNode and EndNode
        elif existing is not node:
            raise ValueError(f'All nodes must have unique node IDs. {node.id!r} was the ID for {existing} and {node}')

    def _insert_edge(self, edge: Edge) -> None:
        assert edge.source_id in self._nodes, f'Edge source {edge.source_id} not found in graph'
        assert edge.destination_id in self._nodes, f'Edge destination {edge.destination_id} not found in graph'
        self._edges_by_source[edge.source_id].append(edge)

    def build(self, parallel: bool = True) -> Graph[StateT, DepsT, GraphInputT, GraphOutputT]:
        # TODO: Warn/error if there is no start node / edges, or end node / edges
        # TODO: Warn/error if the graph is not connected
        # TODO: Warn/error if any non-End node is a dead end
        # TODO: Error if the graph does not meet the every-join-has-a-parent-fork requirement (otherwise can't know when to proceed past joins)
        # TODO: Allow the user to specify the parent forks; only infer them if _not_ specified
        # TODO: Verify that any user-specified parent forks are _actually_ valid parent forks, and if not, generate a helpful error message
        # TODO: Consider doing a deepcopy here to prevent modifications to the underlying nodes and edges
        # TODO: Error if `parallel` is False but the graph has forks / spreads
        nodes = self._nodes
        edges = self._edges_by_source
        nodes, edges = _convert_decision_spreads(nodes, edges)
        parent_forks = _collect_dominating_forks(nodes, edges)

        output_type = cast(type[GraphOutputT], self.output_type)
        if get_origin(output_type) is TypeExpression:
            output_type = get_args(output_type)[0]

        return Graph[StateT, DepsT, GraphInputT, GraphOutputT](
            state_type=self.state_type,
            deps_type=self.deps_type,
            input_type=self.input_type,
            output_type=output_type,
            parallel=parallel,
            nodes=self._nodes,
            edges_by_source=self._edges_by_source,
            parent_forks=parent_forks,
        )


def _convert_decision_spreads(
    graph_nodes: dict[NodeId, AnyNode], graph_edges_by_source: dict[NodeId, list[Edge]]
) -> tuple[dict[NodeId, AnyNode], dict[NodeId, list[Edge]]]:
    def _get_next_spread_id(to: str) -> NodeId:
        prefix = f'spread_to_{to}'
        node_id = prefix
        index = 2
        while node_id in graph_nodes:
            node_id = f'{prefix}_{index}'
            index += 1
        return NodeId(node_id)

    nodes = graph_nodes
    edges = graph_edges_by_source

    for node in list(nodes.values()):
        if isinstance(node, Decision):
            for branch in node.branches:
                if branch.spread:
                    spread = Spread(id=ForkId(_get_next_spread_id(to=branch.route_to.id)))
                    old_route_to = branch.route_to
                    nodes[spread.id] = spread
                    edges[spread.id].append(
                        Edge(
                            source_id=spread.id,
                            transform=branch.post_spread_transform,
                            destination_id=old_route_to.id,
                            user_label=branch.post_spread_user_label,
                        )
                    )
                    branch.route_to = spread
                    branch.spread = False
                    branch.post_spread_transform = None
    return nodes, edges


def _collect_dominating_forks(
    graph_nodes: dict[NodeId, AnyNode], graph_edges_by_source: dict[NodeId, list[Edge]]
) -> dict[JoinId, ParentFork[NodeId]]:
    nodes = set(graph_nodes)
    start_ids = {StartNode.start.id}
    edges = {source_id: [e.destination_id for e in graph_edges_by_source[source_id]] for source_id in nodes}
    for node_id, node in graph_nodes.items():
        if isinstance(node, Decision):
            # For decisions, we need to add edges for the branches
            for branch in node.branches:
                # If any branches have a spread, it's a bug in graph building
                assert not branch.spread, 'Decision branches should not be spreads at this point'
                edges[node_id].append(branch.route_to.id)

    fork_ids = {
        node_id for node_id, node in graph_nodes.items() if isinstance(node, Spread) or len(edges.get(node_id, [])) > 1
    }

    finder = ParentForkFinder(
        nodes=nodes,
        start_ids=start_ids,
        fork_ids=fork_ids,
        edges=edges,
    )

    join_ids = {node.id for node in graph_nodes.values() if isinstance(node, Join)}
    dominating_forks: dict[JoinId, ParentFork[NodeId]] = {}
    for join_id in join_ids:
        dominating_fork = finder.find_parent_fork(join_id)
        if dominating_fork is None:
            # TODO: Print out the mermaid graph and explain the problem
            raise ValueError(f'Join node {join_id} has no dominating fork')
        dominating_forks[join_id] = dominating_fork

    return dominating_forks


@dataclass(repr=False)
class Graph[StateT, DepsT, InputT, OutputT]:
    state_type: type[StateT]
    deps_type: type[DepsT]
    input_type: type[InputT]
    output_type: type[OutputT]

    nodes: dict[NodeId, AnyNode]
    edges_by_source: dict[NodeId, list[Edge]]
    parent_forks: dict[JoinId, ParentFork[NodeId]]

    parallel: bool  # if False, allow direct state modification and don't copy state sent to steps, but disallow parallel node execution

    def __post_init__(self):
        assert StartNode.start.id in self.nodes, 'Graph must have a start node'

    @property
    def start_edges(self) -> list[Edge]:
        return self.edges_by_source.get(START.id, [])

    def get_parent_fork(self, join_id: JoinId) -> ParentFork[NodeId]:
        result = self.parent_forks.get(join_id)
        if result is None:
            raise RuntimeError(f'Node {join_id} is not a join node or did not have a dominating fork (this is a bug)')
        return result

    async def run(self, state: StateT, deps: DepsT, inputs: InputT) -> OutputT:
        return await GraphRun(self, state, deps, inputs).run_async()

    def render(self, *, title: str | None = None, direction: StateDiagramDirection | None = None) -> str:
        return generate_code(self, title=title, direction=direction)

    def __repr__(self):
        return self.render()


WalkerID = uuid.UUID


@dataclass
class GraphWalkState:
    # With our current BaseNode thing, next_node_id and next_node_inputs are merged into `next_node` itself
    node_id: NodeId
    context_inputs: Any
    """
    usually this is the same as node_inputs, but it is different when working with spreads
    """
    node_inputs: Any
    fork_stack: tuple[tuple[ForkId, NodeRunId], ...]
    """
    Stack of forks that have been entered; used so that the GraphRunner can decide when to proceed through joins
    """

    walker_id: WalkerID = field(default_factory=uuid.uuid4)


# TODO: Move `Some` and `Maybe` to util?
@dataclass
class Some[T]:
    value: T


type Maybe[T] = Some[T] | None  # like optional, but you can tell the difference between "no value" and "value is None"

"""
Questions:
* Should you be able to modify state outside of steps and reducer-finalize?
    * E.g., can you modify state in transforms? In reducer input handling? Currently you can access state there..
    * More generally, what do we do about handling state with concurrency?
    * When is state "snapshotted" into persistence? At the start/end of steps/joins, and the whole graph run? Any other time?
    * Is there an option to not persist state at all? It seems yes?
    * Note: it's hard/impossible to resume after an interrupt if you can just make changes to state at any time, because you can't resume from a specific line of code
        * We could allow all active tasks to end before the interrupt goes through, but it feels to me like a reducer-based system might be preferable...

* If the graph run is interrupted, what state needs to be preserved to resume?
    * active reducers, active walks, state, anything else?
    * what needs to happen to concurrently-running tasks if an interruption happens? they get canceled? an error? do we provide some form of interrupt-handling within the context of a larger graph run?
* Should transforms, decisions, and/or reducers be allowed to be async?



Conclusions:
* Everything is sync except step runs
* We allow you to run the graph in a way that state updates are just made on demand, but discourage this (somewhere on the spectrum between docs and runtime error)
when running with parallel execution.
* To allow parallel execution, you need to set `mode='parallel'` as a kwarg to the graph builder, or maybe use a graph subclass or something
    * In parallel mode, you get a runtime error if you try to modify state without using a reducer callback
    * In non-parallel mode, you get a runtime error if you try to create multiple edges from the same node (or create spreads)
    * We recommend updating state using reducer callbacks (recorded on the RunContext) when executing multiple nodes in parallel
* We're not worrying about distributed execution for now, in particular, allowing arbitrary callbacks that modify state in-place as the way we do state updates 
"""


@dataclass
class GraphRunSynchronizer:
    task_group: TaskGroup
    send_stream: MemoryObjectSendStream[tuple[GraphWalkState, Any]]
    receive_stream: MemoryObjectReceiveStream[tuple[GraphWalkState, Any]]
    finish_event: Event


@dataclass(init=False)
class GraphRun[StateT, DepsT, InputT, OutputT]:
    graph: Graph[StateT, DepsT, InputT, OutputT]
    state: StateT
    deps: DepsT
    inputs: InputT

    # persistence: Any  # TODO: Implement use of this

    def __init__(self, graph: Graph[StateT, DepsT, InputT, OutputT], state: StateT, deps: DepsT, inputs: InputT):
        self.graph = graph
        self.state = state
        self.deps = deps
        self.inputs = inputs

        self.result: Maybe[OutputT] = None
        self.active_walks: dict[WalkerID, GraphWalkState] = {}

        self.active_reducers: dict[tuple[NodeId, NodeRunId], Reducer[StateT, DepsT, Any, Any]] = {}
        """The node id is for the join, the node run id is for the dominating fork."""

    async def run_async(self):
        async def handle_finished_steps(
            receive: MemoryObjectReceiveStream[tuple[GraphWalkState, Any]], synch: GraphRunSynchronizer
        ) -> None:
            async with receive:
                async for walk, output in receive:
                    self._end_step(walk, output, synch)

        async with create_task_group() as tg:
            send_result_stream, receive_result_stream = create_memory_object_stream[tuple[GraphWalkState, Any]]()
            finish_event = Event()
            synchronizer = GraphRunSynchronizer(tg, send_result_stream, receive_result_stream, finish_event)
            tg.start_soon(handle_finished_steps, receive_result_stream, synchronizer)

            with send_result_stream:
                start_state = GraphWalkState(
                    node_id=START.id, context_inputs=self.inputs, node_inputs=self.inputs, fork_stack=()
                )
                self._handle_walk(start_state, synchronizer)
                await finish_event.wait()
                tg.cancel_scope.cancel()

        if self.result is None:
            raise RuntimeError(
                'Graph run completed, but no result was produced. This is either a bug in the graph or a bug in the graph runner.'
            )

        return self.result.value

    def _handle_walk(self, walk: GraphWalkState, synchronizer: GraphRunSynchronizer) -> None:
        self.active_walks[walk.walker_id] = walk
        node = self.graph.nodes[walk.node_id]

        if isinstance(node, StartNode):
            self._handle_start(walk, synchronizer)
        elif isinstance(node, Step):
            self._begin_step(node, walk, synchronizer)
            return  # We need to return early here because the `_end_step` call will do the clean-up
        elif isinstance(node, Join):
            self._handle_reduce_join(node, walk)
        elif isinstance(node, Spread):
            self._handle_spread(walk, synchronizer)
        elif isinstance(node, Decision):
            self._handle_decision(node, walk, synchronizer)
        elif isinstance(node, EndNode):
            self._handle_end(walk)

        self._clean_up_walk(walk, synchronizer)

    def _clean_up_walk(self, walk: GraphWalkState, synchronizer: GraphRunSynchronizer) -> None:
        self.active_walks.pop(walk.walker_id)
        self._handle_finalize_joins(walk, synchronizer)
        if not self.active_walks:
            synchronizer.finish_event.set()

    def _handle_start(self, walk: GraphWalkState, synchronizer: GraphRunSynchronizer) -> None:
        # nothing to do besides start the graph
        self._handle_edges(walk, walk.context_inputs, walk.node_inputs, synchronizer)

    def _begin_step(self, step: Step[Any, Any, Any, Any], walk: GraphWalkState, synchronizer: GraphRunSynchronizer):
        step_context = StepContext(self.state, self.deps, walk.context_inputs)

        async def handle_step() -> Any:
            output = await step.call(step_context)
            await synchronizer.send_stream.send((walk, output))

        synchronizer.task_group.start_soon(handle_step)

    def _end_step(self, walk: GraphWalkState, output: Any, synchronizer: GraphRunSynchronizer) -> None:
        self._handle_edges(walk, output, output, synchronizer)
        self._clean_up_walk(walk, synchronizer)

    def _handle_reduce_join(self, join: Join[Any, Any, Any, Any], walk: GraphWalkState) -> None:
        # Find the matching fork run id in the stack; this will be used to look for an active reducer
        parent_fork = self.graph.get_parent_fork(join.id)
        matching_fork_run_id = next(iter(x[1] for x in walk.fork_stack[::-1] if x[0] == parent_fork.fork_id), None)
        if matching_fork_run_id is None:
            raise RuntimeError(
                f'Fork {parent_fork.fork_id} not found in stack {walk.fork_stack}. This means the dominating fork is not dominating (this is a bug).'
            )

        # Get or create the active reducer
        reducer = self.active_reducers.get((join.id, matching_fork_run_id))
        ctx = ReducerContext(self.state, self.deps, walk.node_inputs)

        if reducer is None:
            reducer = join.reducer_factory(ctx)
            self.active_reducers[(join.id, matching_fork_run_id)] = reducer
        else:
            reducer[0](ctx)

    def _handle_spread(self, walk: GraphWalkState, synchronizer: GraphRunSynchronizer):
        self._handle_edges(walk, walk.context_inputs, walk.node_inputs, synchronizer)

    def _handle_decision(
        self, decision: Decision[StateT, DepsT, Any, Any], walk: GraphWalkState, synchronizer: GraphRunSynchronizer
    ) -> None:
        for branch in decision.branches:
            assert not branch.spread, 'Spreads decisions should be converted into spreads as part of graph-building'

            match_tester = branch.matches
            if match_tester is not None:
                inputs_match = match_tester(walk.node_inputs)
            elif branch.source in {Any, object}:
                inputs_match = True
            elif get_origin(branch.source) is Literal:
                inputs_match = walk.node_inputs in get_args(branch.source)
            else:
                inputs_match = isinstance(walk.node_inputs, branch.source)

            if inputs_match:
                node_inputs = walk.node_inputs
                for transform in branch.transforms:
                    ctx = TransformContext(self.state, self.deps, walk.context_inputs, node_inputs)
                    node_inputs = transform(ctx)
                self._handle_walk(
                    GraphWalkState(branch.route_to.id, walk.context_inputs, walk.node_inputs, walk.fork_stack),
                    synchronizer,
                )
                break

    def _handle_end(self, walk: GraphWalkState) -> None:
        self.result = Some(walk.node_inputs)
        # TODO: Probably want to cancel all other walks, terminate the run, etc.

    def _handle_finalize_joins(self, popped_walk: GraphWalkState, synchronizer: GraphRunSynchronizer) -> None:
        # If the popped walk was the last item preventing one or more joins, those joins can now be finalized
        walk_fork_run_ids = {fork_run_id: i for i, (_, fork_run_id) in enumerate(popped_walk.fork_stack)}
        active_reducers_items = list(
            self.active_reducers.items()
        )  # make a copy to avoid modifying the dict while iterating

        # Note: might be more efficient to maintain a better data structure for looking up reducers by join_id and
        # fork_run_id without iterating through every item. This only matters if there is a large number of reducers.
        for (join_id, fork_run_id), reducer in active_reducers_items:
            fork_run_index = walk_fork_run_ids.get(fork_run_id)
            if fork_run_index is not None:
                # This reducer _may_ now be ready to finalize:
                join_can_proceed = True
                for walk in self.active_walks.values():
                    # might be a good idea to hold walks_by_fork_id in memory to reduce overhead here
                    if fork_run_id in {x[1] for x in walk.fork_stack}:
                        join_can_proceed = False

                if join_can_proceed:
                    ctx = ReducerContext(self.state, self.deps, None)
                    output = reducer[1](ctx)
                    new_fork_stack = popped_walk.fork_stack[:fork_run_index]
                    self.active_reducers.pop((join_id, fork_run_id))
                    # Should _now_ traverse the edges leaving this join
                    self._handle_edges(
                        GraphWalkState(join_id, None, None, new_fork_stack), output, output, synchronizer
                    )

    def _handle_edges(
        self, walk: GraphWalkState, context_inputs: Any, next_node_inputs: Any, synchronizer: GraphRunSynchronizer
    ) -> None:
        edges = self.graph.edges_by_source.get(walk.node_id, [])
        node = self.graph.nodes[walk.node_id]

        fork_stack = walk.fork_stack
        if len(edges) > 1 or isinstance(node, Spread):
            # first condition is a broadcast fork; note that the way graph building works,
            # spread nodes should never be broadcast edges; even if there are multiple spreads between two nodes
            # these should result in distinct spread instances
            node_run_id = NodeRunId(str(uuid.uuid4()))
            fork_stack += ((ForkId(walk.node_id), node_run_id),)

        assert not isinstance(node, (Decision, EndNode)), 'This method should not be called for Decision, EndNode'
        if isinstance(node, (StartNode, Step, Join)):
            # Edge transitions should be fast, so maybe don't need to be handled in parallel
            for edge in edges:
                if edge.transform is not None:
                    transform_context = TransformContext(self.state, self.deps, context_inputs, next_node_inputs)
                    next_node_inputs = edge.transform(transform_context)

                self._handle_walk(
                    GraphWalkState(edge.destination_id, context_inputs, next_node_inputs, fork_stack), synchronizer
                )
        elif isinstance(node, Spread):
            for edge in edges:
                for item in next_node_inputs:
                    if edge.transform is not None:
                        transform_context = TransformContext(self.state, self.deps, context_inputs, item)
                        item = edge.transform(transform_context)
                    self._handle_walk(
                        GraphWalkState(edge.destination_id, context_inputs, item, fork_stack), synchronizer
                    )
        else:
            assert_never(node)
