import random
from dataclasses import dataclass
from types import NoneType
from typing import Any, Literal

from pydantic_graph.v2.graph import GraphBuilder, TypeExpression
from pydantic_graph.v2.join import reduce_to_none
from pydantic_graph.v2.step import StepContext


@dataclass
class MyContainer[T]:
    field_1: T | None
    field_2: T | None
    field_3: list[T] | None


@dataclass
class MyState:
    type_name: str | None = None
    container: MyContainer[Any] | None = None


gb = GraphBuilder(state_type=MyState, deps_type=NoneType, input_type=NoneType, output_type=NoneType)


@gb.step
async def choose_type(ctx: StepContext[MyState, object, object]) -> Literal['int', 'str']:
    chosen_type = int if random.random() < 0.5 else str
    ctx.state.type_name = chosen_type.__name__
    ctx.state.container = MyContainer(field_1=None, field_2=None, field_3=None)
    return 'int' if chosen_type is int else 'str'


@gb.step
async def handle_int(ctx: StepContext[object, object, object]) -> None:
    pass


@gb.step
async def handle_str(ctx: StepContext[object, object, object]) -> None:
    pass


@gb.step
async def handle_int_1(ctx: StepContext[MyState, object, object]) -> None:
    print('start int 1')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_1 = 1
    print('end int 1')


@gb.step
async def handle_int_2(ctx: StepContext[MyState, object, object]) -> None:
    print('start int 2')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_2 = 1
    print('end int 2')


@gb.step
async def handle_int_3(ctx: StepContext[MyState, object, object]) -> None:
    print('start int 3')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_3 = []
    print('end int 3')


@gb.step
async def handle_str_1(ctx: StepContext[MyState, object, object]) -> None:
    print('start str 1')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_1 = 1
    print('end str 1')


@gb.step
async def handle_str_2(ctx: StepContext[MyState, object, object]) -> None:
    print('start str 2')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_2 = 1
    print('end str 2')


@gb.step
async def handle_str_3(ctx: StepContext[MyState, object, object]) -> None:
    print('start str 3')
    await asyncio.sleep(1)
    assert ctx.state.container is not None
    ctx.state.container.field_3 = []
    print('end str 3')


handle_int_join = gb.join(reduce_to_none, node_id='handle_int_join')
handle_str_join = gb.join(reduce_to_none, node_id='handle_str_join')

gb.start_with(choose_type, label='begin')

gb.add_edge(
    (h := gb.get_handler(choose_type)).source,
    gb.decision()
    .branch(h(TypeExpression[Literal['int']]).route_to(handle_int))
    .branch(h(TypeExpression[Literal['str']]).route_to(handle_str)),
)
# If you don't care about type-checking the inputs, you can just use:
# gb.add_edge(
#     choose_type,
#     gb.decision()
#     .branch(gb.handle(TypeExpression[Literal['int']]).route_to(handle_int))
#     .branch(gb.handle(TypeExpression[Literal['str']]).route_to(handle_str))
# )


for handle_int_field in (handle_int_1, handle_int_2, handle_int_3):
    gb.add_edge(handle_int, handle_int_field)
    gb.add_edge(handle_int_field, handle_int_join)
gb.end_from(handle_int_join)

for handle_str_field in (handle_str_1, handle_str_2, handle_str_3):
    gb.add_edge(handle_str, handle_str_field)
    gb.add_edge(handle_str_field, handle_str_join)
gb.end_from(handle_str_join)

g = gb.build()
print(g)
print('-')


async def main():
    state = MyState()
    result = await g.run(state=state, deps=None, inputs=None)
    print('-')
    print(f'{result=}')
    print(f'{state=}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
