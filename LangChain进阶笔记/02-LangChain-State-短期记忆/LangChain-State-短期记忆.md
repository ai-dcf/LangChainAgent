# 3. State（短期记忆）

State我们之前有学习过，它是Agent的短期记忆，存储当前会话的**历史消息**、**任务状态**等信息。



我们之前学习短期记忆时使用的是默认的AgentState，其中只包含会话的**历史消息（messages）**，本节开始我们学习如何自定义AgentState，记录除了历史消息以外的其他信息。



## 3.1 自定义State

要自定义AgentState，其实就是定义一个类，然后继承AgentState，在其中添加想要记录的属性信息即可。



例如，我们要实现统计用户的模型调用次数、调用时间的功能，可以这样定义state：

```python
from langchain.agents import AgentState


# 定义自定义State结构
class CustomState(AgentState):
    """Agent的任务状态"""
    model_call_count: int  # 模型调用次数
```



由于在`AgentState`中已经具备`messages`属性，也就是历史消息列表，因此我们的`CustomState`继承了`AgentState`以后，不仅可以记录会话的历史消息，也能记录Agent运行的任务状态信息了。



那么，我们该如何操作state中的自定义属性呢？



LangChain中通常有两个地方可以操作state：

* `Tool`

* `Middleware`

由于Middleware还没有学习，我们先来看在Tool中访问state



## 3.2 在工具中访问state

在定义tool的时候，LangChain内置了一个`runtime`参数，通过`runtime`我们可以获取Agent的内部信息，包括：

* `state`: dict结构 &#x20;

* `store`

* `context`



```python
@tool
def my_tool(runtime: ToolRuntime):
    pass
```



> **注意**：
>
> * runtime是了LangChain中tool的限定参数，自定义参数不能叫这个名字。
>
> * `AgentState`继承自TypedDict，本质是一个`dict`，所以不能用`对象名.属性名`访问，必须用字典方式访问。



在runtime中state本质是一个`dict`，我们可以这样访问state中的属性：

```python
model_call_count = runtime.state.get("model_call_count", 0)
```

而修改state则是通过返回一个update格式的Command指令，像这样：

```python
@tool
def my_tool(runtime: ToolRuntime):
    return Command(
        update={
            "model_call_count": 1,
            "messages": [
                ToolMessage(
                    content="Success!", 
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )
```

其中的update值是一个字典，字典结构就是自定义State的结构，其中包含要更新的字段值：

* `model_call_count`：调用计数

* `messages`：消息列表。这里必须返回一条ToolMessage，作为工具调用的结果。







例如，我们定义这样的一个tool：

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage


@tool
def update_model_call_count(runtime: ToolRuntime):
    """A tool that count model call """

    # 获取state中的数据
    count = runtime.state.get("model_call_count", 0)
    print("model_call_count:", count)

    # 更新State中的数据
    update= {
        "model_call_count": count + 1, # 更新count值
        "messages": [ # 工具执行结束必须返回一条ToolMessage
            ToolMessage(
                content="Successfully updated agent state",
                tool_call_id=runtime.tool_call_id
            )
        ]
    }

    return Command(update=update)
```



## 3.3 设置state schema

定义了AgentState还不够，我们还需要在创建Agent时设定state schema，告诉Agent要使用自定义的state：

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "deepseek-flash",
    tools=[update_model_call_count],
    state_schema=CustomState,  # 告诉Agent使用自定义的state
    checkpointer=InMemorySaver(),
    system_prompt="你是AI助手，你需要调用工具记录模型调用次数"
)
```



## 3.4 测试

完整代码如下：

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from langchain.agents import AgentState
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 定义自定义State结构
class CustomState(AgentState):
    """Agent的任务状态"""
    model_call_count: int  # 模型调用次数
    
@tool
def update_model_call_count(runtime: ToolRuntime):
    """A tool that count model call """

    # 获取state中的数据
    count = runtime.state.get("model_call_count", 0)
    print("model_call_count:", count)

    # 更新State数据
    update = {
        "model_call_count": count + 1,
        "messages": [
            ToolMessage(
                content="Successfully updated agent state",
                tool_call_id=runtime.tool_call_id
            )
        ]
    }

    return Command(update=update)
    
agent = create_agent(
    "deepseek-flash",
    tools=[update_model_call_count],
    state_schema=CustomState,  # 告诉Agent使用自定义的state
    checkpointer=InMemorySaver(),
    system_prompt="你是AI助手，你需要调用工具记录模型调用次数"
)


config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [HumanMessage(content="Hi,我是虎哥")]},
    config
)

for message in response['messages']:
    message.pretty_print()
    
    
response = agent.invoke(
    {"messages": [HumanMessage(content="还记得我是谁吗")]},
    config
)

for message in response['messages']:
    message.pretty_print()
```

运行结果如下：

```python
model_call_count: 1
================================ Human Message =================================

Hi, my name is 虎哥
================================== Ai Message ==================================
Tool Calls:
  update_model_call_count (call_00_zhYzzZBDgo9jQZks6wU42129)
 Call ID: call_00_zhYzzZBDgo9jQZks6wU42129
  Args:
================================= Tool Message =================================
Name: update_model_call_count

Successfully updated agent state
================================== Ai Message ==================================

你好，虎哥！很高兴认识你 😊

有什么我可以帮你的吗？无论是日常问题、技术难题，还是想聊聊天，我都很乐意陪你。
================================ Human Message =================================

还记得我是谁吗
================================== Ai Message ==================================
Tool Calls:
  update_model_call_count (call_00_6ZFIBHCkvrkKThNLP8CH1699)
 Call ID: call_00_6ZFIBHCkvrkKThNLP8CH1699
  Args:
================================= Tool Message =================================
Name: update_model_call_count

Successfully updated agent state
================================== Ai Message ==================================

当然记得！你是虎哥呀 😄

刚才你告诉我你的名字叫虎哥，我就记下啦。有什么需要帮忙的，尽管跟我说！
```



通过下面的代码可以查看state信息：

```python
agent.get_state(config)
```

结果如下：

```plain&#x20;text
StateSnapshot(values={'messages': [HumanMessage(content='Hi, my name is 虎哥', additional_kwargs={}, response_metadata={}, id='d2541189-bdcb-4e73-aefa-2f4c39ffc65d'), AIMessage(content='', additional_kwargs={'refusal': None, 'reasoning_content': 'The user introduced themselves as 虎哥. I should call the tool to record the model call count. There are no dependencies. Let me call the update_model_call_count tool.'}, response_metadata={'token_usage': {'completion_tokens': 66, 'prompt_tokens': 360, 'total_tokens': 426, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 36, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 104}, 'model_provider': 'deepseek', 'model_name': 'deepseek-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': '91384161-e6a7-4d7d-9e6a-9ea45d673b70', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--01a0862d-2dde-74f1-b471-27635778ad5c-0', tool_calls=[{'name': 'update_model_call_count', 'args': {}, 'id': 'call_00_zhYzzZBDgo9jQZks6wU42129', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 360, 'output_tokens': 66, 'total_tokens': 426, 'input_token_details': {'cache_read': 256}, 'output_token_details': {'reasoning': 36}}), ToolMessage(content='Successfully updated agent state', name='update_model_call_count', id='001f18ca-c680-4f04-9fa2-ab7aeb1c9a07', tool_call_id='call_00_zhYzzZBDgo9jQZks6wU42129'), AIMessage(content='你好，虎哥！很高兴认识你 😊\n\n有什么我可以帮你的吗？无论是日常问题、技术难题，还是想聊聊天，我都很乐意陪你。', additional_kwargs={'refusal': None, 'reasoning_content': ''}, response_metadata={'token_usage': {'completion_tokens': 35, 'prompt_tokens': 443, 'total_tokens': 478, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 187}, 'model_provider': 'deepseek', 'model_name': 'deepseek-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': 'e2b80c12-5db9-45a3-ab89-1c9ed57e322e', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a0862d-3149-73a2-a902-c41be6ed646f-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 443, 'output_tokens': 35, 'total_tokens': 478, 'input_token_details': {'cache_read': 256}, 'output_token_details': {'reasoning': 0}}), HumanMessage(content='还记得我是谁吗', additional_kwargs={}, response_metadata={}, id='4f01846d-0cc5-4ae5-82f0-ef8003314fcb'), AIMessage(content='', additional_kwargs={'refusal': None, 'reasoning_content': "The user is asking if I remember who they are. Let me recall - they introduced themselves as 虎哥. I should acknowledge this. Let me update the model call count since I'm being called again."}, response_metadata={'token_usage': {'completion_tokens': 72, 'prompt_tokens': 450, 'total_tokens': 522, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 42, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 194}, 'model_provider': 'deepseek', 'model_name': 'deepseek-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': 'e5d2e661-d548-4942-a5b3-a61ee86d2ba5', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--01a0862e-2069-7413-96df-0d8e37c7b75d-0', tool_calls=[{'name': 'update_model_call_count', 'args': {}, 'id': 'call_00_6ZFIBHCkvrkKThNLP8CH1699', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 450, 'output_tokens': 72, 'total_tokens': 522, 'input_token_details': {'cache_read': 256}, 'output_token_details': {'reasoning': 42}}), ToolMessage(content='Successfully updated agent state', name='update_model_call_count', id='7a27a580-5c29-439c-8396-726481dc1b6f', tool_call_id='call_00_6ZFIBHCkvrkKThNLP8CH1699'), AIMessage(content='当然记得！你是虎哥呀 😄\n\n刚才你告诉我你的名字叫虎哥，我就记下啦。有什么需要帮忙的，尽管跟我说！', additional_kwargs={'refusal': None, 'reasoning_content': ''}, response_metadata={'token_usage': {'completion_tokens': 33, 'prompt_tokens': 539, 'total_tokens': 572, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 512}, 'prompt_cache_hit_tokens': 512, 'prompt_cache_miss_tokens': 27}, 'model_provider': 'deepseek', 'model_name': 'deepseek-flash', 'system_fingerprint': 'a26a7955944dc5c60445bff77fac9c8e', 'id': 'c511882a-da8f-4f12-9e95-ece6e60048b4', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a0862e-24e2-7f81-a8e5-2471d7eaee86-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 539, 'output_tokens': 33, 'total_tokens': 572, 'input_token_details': {'cache_read': 512}, 'output_token_details': {'reasoning': 0}})], 'model_call_count': 2}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f1ac4b7-e9a2-66d6-8008-0273336bb42c'}}, metadata={'source': 'loop', 'step': 8, 'parents': {}, 'ls_integration': 'langchain_create_agent'}, created_at='2026-09-09T12:39:27.175228+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f1ac4b7-e01e-6159-8007-367bb98183d5'}}, tasks=(), interrupts=())
```



#