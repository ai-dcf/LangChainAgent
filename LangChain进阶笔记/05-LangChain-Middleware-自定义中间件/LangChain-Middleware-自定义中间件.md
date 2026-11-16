# 2. 自定义中间件

除了使用预置中间件，我们还可以利用LangChain提供的hook创建完全自定义的中间件。



根据hook的种类，中间件可以分为两类：

* Node-style hooks：在具体某个节点执行的中间件，包含：

  * before\_agent

  * before\_model

  * after\_model

  * after\_agent

* Wrap-style hooks：环绕model或tool调用的中间件，包括：

  * wrap\_model\_call

  * wrap\_tool\_call

为了便于开发中间件，LangChain为每一种hook都提供了装饰器，我们只要定义函数并使用装饰器标记即可快速开发中间件。



## 2.1 node-style装饰器

* Node-style hooks：在具体某个节点执行的中间件，包含：

  * before\_agent

  * before\_model

  * after\_model

  * after\_agent



例如，我们要实现模型调用计数功能，之前利用tool来统计增加了很多次tool调用，浪费token，而且也不太方便。利用中间件就优雅很多。我们可以在每次调用模型后(**after\_model**)记录模型调用次数。



做法是定义函数，并用`@after_model`装饰器来装饰该函数，但要注意，**函数的参数和返回值必须严格按照下面的示例**：

```python
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from langchain.agents.middleware import after_model


# 1.定义自定义state，记录模型调用次数
class CustomState(AgentState):
    """扩展Agent状态，添加自定义字段"""
    model_call_count: int  # 模型调用次数


# 2.定义中间件
@after_model(state_schema=CustomState)
def increment_counter(
    state: CustomState,
    runtime: Runtime
):

    # 读取State中的count值
    current_count = state.get("model_call_count", 0)
    # 返回dict，其中的key/value是要更新的State值
    return {"model_call_count": current_count + 1}
```

测试：

```python
# 创建智能体，设置middleware
agent = create_agent(
    model="deepseek-chat",
    middleware=[increment_counter],
    checkpointer=InMemorySaver()
)
config = {"configurable": {"thread_id": "1"}}

# 调用智能体
response = agent.invoke(
    {"messages": [HumanMessage("Hello")],},
    config
)

print(response)
```

结果：

```python
{'messages': [HumanMessage(content='Hello', additional_kwargs={}, response_metadata={}, id='f333c2ff-d50b-452c-941b-adb1c69812f6'), AIMessage(content='你好！有什么可以帮你的吗？😊', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 5, 'total_tokens': 15, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 5}, 'model_provider': 'deepseek', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'fp_058df29938_prod0820_fp8_kvcache_20260402', 'id': 'fc0a7204-aafb-44f2-89a0-c516096134b1', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019dbe9e-f818-7b82-86ab-dd39577897b2-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 5, 'output_tokens': 10, 'total_tokens': 15, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})], 'model_call_count': 1}
```







## 2.2 wrap-style装饰器

* Wrap-style hooks：环绕model或tool调用的中间件，包括：

  * wrap\_model\_call

  * wrap\_tool\_call



例如，我们来定义一个可以在调用模型时失败重试的中间件，最大重试次数为3次。那就可以使用`@wrap_model_call`，在模型调用时做出判断，如果失败则重试，重试此时超过3次则结束。



同样的，被`@wrap_model_call`装饰的函数，其参数和返回值必须严格按照下面的格式：

```python
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from typing import Any, Callable


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request) # 这一步就是调用模型
        except Exception as e:
            print(f"Retry {attempt + 1}/3 after error: {e}")
            if attempt == 2:
                raise
    return None


agent = create_agent(
    model="deepseek-chat",
    middleware=[retry_model]
)
```



测试：

```python
config = {"configurable": {"thread_id": "1"}}

# 调用智能体，并初始化state
response = agent.invoke(
    {"messages": [HumanMessage("Hello")]},
    config,
)

print(response)
```



## 2.3 类装饰器

对于需要同时用到多个hooks的更复杂的中间件逻辑，我们还可以使用自定义类继承AgentMiddleware的方式来创建中间件。

例如，一个用来记录日志的中间件，要在模型调用、工具调用前后记录日志：

```python
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ToolCallRequest
from langgraph.types import Command


class LoggingMiddleware(AgentMiddleware):

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        try:
            print(f"\n=======About to call model with {len(request.messages)} messages=======")
            return handler(request)
        except Exception as e:
            print(f"\n=======[错误]: {str(e)}=======")
            return AIMessage("调用模型失败，请重试~")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        print(f"\n=======调用工具: {request.tool_call['name']}=======")
        print(f"\n=======参数: {request.tool_call['args']}=======")
        try:
            result = handler(request)
            print("\n=======工具调用成功！=======")
            return result
        except Exception as e:
            print(f"\n=======工具调用失败: {e}=======")
            raise


@tool
def get_weather(location: str):
    """查询指定城市的天气信息"""
    return f"Current weather in {location} is sunny, 25℃."


agent = create_agent(
    model="deepseek-chat",
    middleware=[LoggingMiddleware()],
    tools=[get_weather],
)

for chunk, metadata in agent.stream(
    {"messages": [HumanMessage("杭州今天天气如何？")]},
    stream_mode="messages"
):
    if chunk and chunk.content:
        print(chunk.content, end="", flush=True)
```

运行结果：

```sql
=======About to call model with 1 messages=======
好的，我来查询一下杭州今天的天气情况。
=======调用工具: get_weather=======

=======参数: {'location': '杭州'}=======

=======工具调用成功！=======
Current weather in 杭州 is sunny, 25℃.
=======About to call model with 3 messages=======
杭州今天天气晴朗，气温 **25℃**，非常适合外出活动哦！🌞 注意适当防晒～
```





# 3. 高级用法

中间件除了利用hook做基本的信息记录和判断，还可以有一些高级的用法，例如：

* 动态修改请求：可以拦截发送给模型的请求，动态修改请求中使用的模型、工具、提示词等

* 条件跳转：在满足条件的情况下直接跳转到某个Agent执行的节点





## 3.1 动态修改请求参数

在wrap\_model\_call这个hook中，我们可以动态修改任意的request参数，包括：

* model

* tool

* system\_prompt

* ...

关键点是利用`request.override()`来覆盖原本的request参数。



例如，我们定义中间件，可以根据context中用户的选择来开启或关闭模型的思考模式。

```python
from langchain.agents.middleware import wrap_model_call
from collections.abc import Callable
from pydantic.dataclasses import dataclass

# 定义一个context，记录用户是否开启思考模式
@dataclass
class UserContext:
    thinking: bool = False


@wrap_model_call
def dynamic_model_settings(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:

    # 获取上下文参数thinking，是否开启思考模式
    thinking = request.runtime.context.thinking

    # 如果Thinking为True，则开启思考模式，否则关闭思考模式
    thinking_type = "enabled" if thinking else "disabled"
    model_settings = {"extra_body": {"thinking": {"type": thinking_type}}}

    # 覆盖request中的model参数
    modified_request = request.override(model_settings=model_settings)
    return handler(modified_request)
```



测试：

```python
# 创建智能体，设置middleware
agent = create_agent(
    model="deepseek-flash",
    middleware=[dynamic_model_settings],
    context_schema=UserContext
)

# 调用智能体，并初始化state
response = agent.invoke(
    {"messages": [HumanMessage("Hello")]},
    context=UserContext(thinking=False) # 通过Context控制思考模式开关
)

print(response)
```





## 3.2 条件跳转

在中间件使用`jump_to`指令可以跳过模型调用，直接进入指定的Agent节点。



例如，我们设定一个模型限流的中间件，当判断用户模型调用次数超过阈值，直接禁止调用，结束Agent运行。

```python
from langgraph.runtime import Runtime
from typing import NotRequired, Any
from langchain.agents.middleware import AgentMiddleware, hook_config, AgentState


# 1.定义自定义state，记录模型调用次数
class CustomAgentState(AgentState):
    """扩展Agent状态，添加自定义字段"""
    model_call_count: NotRequired[int]  # 模型调用次数


# 2.定义中间件
class ModelCallLimitMiddleware(AgentMiddleware[CustomAgentState]):

    def __init__(self, max_limit: int = 10):
        super().__init__()
        self.max_limit = max_limit

    # after_model hook，模型调用次数计数
    def after_model(self, state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
        current_count = state.get("model_call_count", 0)
        return {"model_call_count": current_count + 1}

    # before_model hook，校验模型调用次数是否超限
    @hook_config(can_jump_to=["end"])
    def before_model(self, state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
        # 获取模型调用次数
        current_count = state.get("model_call_count", 0) + 1
        # 判断是否超限
        print(f"当前模型调用次数：{current_count}/{self.max_limit}")
        if current_count > self.max_limit:
            return {
                "jump_to": "end",
                "messages": AIMessage(f"模型调用次数超过最大限制：{self.max_limit}！")
            }
        return None

# @before_model(can_jump_to=["end"])
# def check_call_limit(state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
#     pass
```

测试：

```python
# 创建智能体，设置middleware
agent = create_agent(
    model="deepseek-chat",
    middleware=[ModelCallLimitMiddleware(max_limit=2)],
    checkpointer=InMemorySaver(),
    state_schema=CustomAgentState
)
config = {"configurable": {"thread_id": "1"}}

# 调用智能体，并初始化state
response = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
}, config)

print(response)
```

&#x20;

