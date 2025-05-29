from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast, get_args, get_origin

from pydantic import TypeAdapter
from pydantic_core import to_json
from pydantic_graph.v2.id_types import NodeId
from pydantic_graph.v2.step import StepContext
from pydantic_graph.v2.util import TypeExpression

from pydantic_ai import Agent, models


@dataclass
class Prompt[InputT, OutputT]:
    input_type: type[InputT]
    output_type: type[TypeExpression[OutputT]] | type[OutputT]
    prompt: str
    model: models.Model | models.KnownModelName | str = 'openai:gpt-4o'

    @cached_property
    def agent(self) -> Agent[None, OutputT]:
        input_json_schema = to_json(
            TypeAdapter(self.input_type).json_schema(), indent=2
        ).decode()
        instructions = '\n'.join(
            [
                'You will receive messages matching the following JSON schema:',
                input_json_schema,
                '',
                'Generate output based on the following instructions:',
                self.prompt,
            ]
        )
        output_type = self.output_type
        if get_origin(output_type) is TypeExpression:
            output_type = get_args(self.output_type)[0]
        return Agent(
            model=self.model,
            output_type=cast(type[OutputT], output_type),
            instructions=instructions,
        )

    async def __call__(self, ctx: StepContext[Any, Any, InputT]) -> OutputT:
        result = self.agent.run_sync(to_json(ctx.inputs, indent=2).decode())
        return result.output


@dataclass
class Interruption[StopT, ResumeT]:
    value: StopT
    next_node: (
        NodeId  # This is the node this walk should resume from after the interruption
    )
    graph_state: Any = None  # TODO: Need a way to pass the graph state ...?
