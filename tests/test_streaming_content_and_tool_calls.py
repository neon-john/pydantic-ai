from collections.abc import AsyncIterator, Callable

import pytest

from pydantic_ai import (
    Agent,
    ModelRetry,
    Tool,
    UnexpectedModelBehavior,
    capture_run_messages,
    models,
)
from pydantic_ai.messages import (
    ModelMessage,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.openai import OpenAIModel

models.ALLOW_MODEL_REQUESTS = False

model = OpenAIModel(model_name='gpt-4o', api_key='foobar')


class CustomException(Exception):
    pass


async def always_retry_tool(input: str) -> str:
    raise ModelRetry(f'Try again: {input}')


async def always_throw_custom_exception_tool(input: str) -> str:
    raise CustomException(f'Something went wrong: {input}')


async def echo_tool(input: str) -> str:
    return input


tools = [
    Tool(always_retry_tool, name='always_retry_tool'),
    Tool(
        always_throw_custom_exception_tool,
        name='always_throw_custom_exception_tool',
    ),
    Tool(echo_tool, name='echo_tool'),
]

pytestmark = pytest.mark.anyio


# Pydantic AI agent
agent = Agent(
    model=model,
    retries=2,
    tools=tools,
)


def create_stream_function(
    function_tool_name: str | None = None,
    text_and_tool_call: bool = False,
    text_only: bool = False,
) -> Callable[[list[ModelMessage], AgentInfo], AsyncIterator[str | DeltaToolCalls]]:
    async def model_stream_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        if text_only:
            yield 'stream complete'
            return

        input = f'{{"input": "{len(messages)}"}}'
        if len(messages) == 1:  # request the tool call
            if text_and_tool_call:
                yield 'Calling a tool.'

            yield {0: DeltaToolCall(name=function_tool_name)}
            yield {0: DeltaToolCall(json_args=input)}
        elif messages[-1].parts[0].part_kind == 'retry-prompt':  # subsequent messages (retries)
            if text_and_tool_call:  # yield error message if requested
                yield "I'm sorry! Something went wrong, but let me try that again."

            # always request tool retry
            yield {0: DeltaToolCall(name=function_tool_name)}
            yield {0: DeltaToolCall(json_args=input)}
        else:
            yield 'stream tool call complete'

    return model_stream_function


async def run_stream(agent: Agent, model: FunctionModel) -> tuple[list[ModelMessage] | None, str, Exception | None]:
    exception = None
    streamed = ''
    with capture_run_messages() as messages:
        with agent.override(model=model):
            try:
                async with agent.run_stream('Do something') as response:
                    async for chunk in response.stream_text():
                        streamed += chunk
            except Exception as e:
                exception = e

    return messages, streamed, exception


async def test_always_retry_tool() -> None:
    """Call `always_retry_tool`.

    Should return 6 messages:
        * [ModelRequest] Prompt
        * [ModelResponse] Tool Request
        * [ModelRequest] Provide tool error details
        * [ModelResponse] Tool Request
        * [ModelRequest] Provide tool error details
        * [ModelResponse] Tool Request
    Cannot retry again, exception thrown
    """
    function_model = FunctionModel(stream_function=create_stream_function('always_retry_tool'))

    messages, _, exception = await run_stream(agent, function_model)
    assert isinstance(exception, UnexpectedModelBehavior)
    assert messages is not None
    assert len(messages) == 6
    assert isinstance(messages[5].parts[0], ToolCallPart)

    messages, _, exception = await run_stream(agent, function_model)
    assert isinstance(exception, UnexpectedModelBehavior)
    assert messages is not None
    assert len(messages) == 6
    assert isinstance(messages[5].parts[0], ToolCallPart)


async def test_always_throw_tool() -> None:
    """Call `always_throw_custom_exception_tool`.

    Should return 2 messages:
        * [ModelRequest] Prompt
        * [ModelResponse] Tool Request
    Custom exception thrown
    """
    function_model = FunctionModel(stream_function=create_stream_function('always_throw_custom_exception_tool'))

    messages, _, exception = await run_stream(agent, function_model)
    assert isinstance(exception, CustomException)
    assert messages is not None
    assert len(messages) == 2
    assert isinstance(messages[1].parts[0], ToolCallPart)

    messages, _, exception = await run_stream(agent, function_model)
    assert isinstance(exception, CustomException)
    assert messages is not None
    assert len(messages) == 2
    assert isinstance(messages[1].parts[0], ToolCallPart)


async def test_echo_tool() -> None:
    """Call `echo_tool`.

    Should return 4 messages:
        * [ModelRequest] Prompt
        * [ModelResponse] Tool Request
        * [ModelRequest] Provide tool result
        * [ModelResponse] 'stream tool call complete'
    """
    function_model = FunctionModel(stream_function=create_stream_function('echo_tool'))

    messages, _, _ = await run_stream(agent, function_model)
    assert messages is not None
    assert len(messages) == 4
    assert isinstance(messages[3].parts[0], TextPart)
    assert messages[3].parts[0].content == 'stream tool call complete'

    messages, _, _ = await run_stream(agent, function_model)
    assert messages is not None
    assert len(messages) == 4
    assert isinstance(messages[3].parts[0], TextPart)
    assert messages[3].parts[0].content == 'stream tool call complete'


async def test_no_tool_calls() -> None:
    """Call no tools.

    Should return 2 messages:
        * [ModelRequest] Prompt
        * [ModelResponse] 'stream complete'
    """
    function_model = FunctionModel(stream_function=create_stream_function(text_only=True))

    messages, _, _ = await run_stream(agent, function_model)
    assert messages is not None
    assert len(messages) == 2
    assert type(messages[1].parts[0]) is TextPart
    assert messages[1].parts[0].content == 'stream complete'

    messages, _, _ = await run_stream(agent, function_model)
    assert messages is not None
    assert len(messages) == 2
    assert type(messages[1].parts[0]) is TextPart
    assert messages[1].parts[0].content == 'stream complete'


async def test_always_retry_tool_with_text_and_tool_call() -> None:
    """Call always_retry_tool w/ text_and_tool_call response.

    Should return 6 messages:
        * [ModelRequest] Prompt
        * [ModelResponse] Tool Request (w/ 'Calling a tool.' streaming)
        * [ModelRequest] Provide tool error details,
        * [ModelResponse] Tool Request  (w/ 'I'm sorry! Something went wrong, but let me try that again.' streaming)
        * [ModelRequest] Provide tool error details,
        * [ModelResponse] Tool Request  (w/ 'I'm sorry! Something went wrong, but let me try that again.' streaming)
    Cannot retry again, exception thrown
    """
    function_model = FunctionModel(stream_function=create_stream_function('always_retry_tool', text_and_tool_call=True))

    messages, streamed, exception = await run_stream(agent, function_model)
    assert isinstance(exception, UnexpectedModelBehavior)
    assert messages is not None
    assert len(messages) == 6
    assert streamed == 'Calling a tool.' + "I'm sorry! Something went wrong, but let me try that again." * 2


async def test_always_throw_tool_with_text_and_tool_call() -> None:
    """Call `always_throw_custom_exception_tool`.

    Should return 2 messages:
        * [ModelRequest] Prompt
        * [ModelResponse] Tool Request (w/ 'Calling a tool.' streaming)
    Custom exception thrown
    """
    function_model = FunctionModel(
        stream_function=create_stream_function('always_throw_custom_exception_tool', text_and_tool_call=True)
    )

    messages, streamed, exception = await run_stream(agent, function_model)
    assert isinstance(exception, CustomException)
    assert messages is not None
    assert len(messages) == 2
    assert streamed == 'Calling a tool.'


async def test_echo_tool_with_text_and_tool_call() -> None:
    """Call `echo_tool`.

    Should return 4 messages:
        * [ModelRequest] Prompt
        * [ModelResponse] Tool Request (w/ 'Calling a tool.' streaming)
        * [ModelRequest] Provide tool result
        * [ModelResponse] 'stream tool call complete'
    """
    function_model = FunctionModel(stream_function=create_stream_function('echo_tool', text_and_tool_call=True))

    messages, streamed, exception = await run_stream(agent, function_model)
    assert exception is None
    assert messages is not None
    assert len(messages) == 4
    assert streamed == 'Calling a tool.stream tool call complete'
