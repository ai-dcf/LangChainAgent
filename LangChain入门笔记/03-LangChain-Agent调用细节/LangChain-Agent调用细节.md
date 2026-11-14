# 4. Agent调用细节

## 4.1 调用Agent

创建Agent的基本流程是这样的：

* 1.准备并加载环境变量

* 2.初始化模型

* 3.利用`@tool`定义工具

* 4.定义Agent，绑定model和tools

* 5.调用Agent



而调用Agent也分为两种模式：

* `invoke`：阻塞式调用

* `stream_events`：流式调用



### 4.1.1 invoke

完整代码：

```python
# invoke调用Agent
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 1.加载环境变量
load_dotenv()

# 2.初始化模型
model = init_chat_model(
        "deepseek-v4-flash",
        extra_body={"thinking": {"type": "disabled"}}
    )

# 3.定义tool（可以没有，不过这个时候只能对话，不算完整的Agent）
tools=[]

# 4.创建智能体
agent = create_agent(
    model = model,
    tools=tools
)

# 5.调用Agent
response = agent.invoke({
    "messages": [HumanMessage("你是谁？")]
})
print(response['messages'][-1].content)
```



注意Agent同步调用与Model同步调用的差别：

* 传入参数不同

  * 调用模型：str | list，单个消息或消息的列表

  * 调用Agent：dict，key为messages，value为消息列表

* 返回值不同

  * 调用模型：返回值是AIMessage

  * 调用Agent：返回值是Dict，其中的'messages'是所有历史消息



> **总结**：
>
> * **Model调用**，传入Messages(一条或多条），返回的也是Message（AIMessage）
>
> * **Agent调用**，传入dict，返回也是dict，其中都包含messages，AI返回结果再messages最后一条



### 4.1.2 stream\_events

LangChain的1.3版本之后推出了新的stream模式，称为"V3"，调用函数&#x662F;**`stream_events`**，返回值一&#x4E2A;**`GraphRunStream`**&#x7C7B;，你可以自己选择想要通过stream获取的内容：

* **stream** : 包含每个协议事件.

  * **stream.messages** : 模型调用的消息流（message stream），可以遍历得到message

    * message.text：消息片段和最终完整消息

  * **stream.values** :   执行过程中的Agent的state快照.

  * **stream.output** :   最终的Agent state.

  * **stream.subgraphs** :   子Agent运行信息（多Agent嵌套时用）.

  * **stream.extensions** : 自定义事件流.



示例代码：

```python
# stream_events 调用Agent
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 1.加载环境变量
load_dotenv()

# 2.初始化模型
model = init_chat_model(
        "deepseek-v4-flash",
        extra_body={"thinking": {"type": "disabled"}}
    )

# 3.定义tool（可以没有，不过这个时候只能对话，不算完整的Agent）
tools=[]

# 4.创建智能体
agent = create_agent(
    model = model,
    tools=tools
)

# 5.流式调用
stream = agent.stream_events(
    {"messages": [HumanMessage(content="你是谁？")]},
    version="v3"
)

for message in stream.messages:
    for text in message.text:
        print(text, end="", flush=True)
```



> **stream\_events调用参数说明**：
>
> * 参数1：dict，key为messages，value为消息列表，这个与`invoke`调用一致
>
> * 参数2：stream的API版本，这里推荐V3
>
> 
>
> **stream\_events响应结果说明：**
>
> * 返回的stream中包含各种event stream，其中`stream.messages`是获取message的stream
>
> * `message.text`是AI的消息片段的str文本





## 4.2 系统提示词（Prompts）

发送给大模型的所有消息都可以称为**提示词（Prompt）**，它直接影响模型的输出结&#x679C;**。**



其中，SystemMessage尤为重要，我们把SystemMessage称为**系统提示词**（**System Prompt**），它可以给模型设定角色和本次聊天的背景，对模型生成的内容有很大的影响。



在创建智能体时，我们可以直接设定system prompt，不必在每次发送消息时指定。

```python
# 系统提示词
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 1.加载环境变量
load_dotenv()

# 2.初始化模型
model = init_chat_model(
        "deepseek-v4-flash",
        extra_body={"thinking": {"type": "disabled"}}
    )

# 3.定义tool（可以没有，不过这个时候只能对话，不算完整的Agent）
tools=[]

# 4.创建智能体
agent = create_agent(
    model = model,
    tools=tools,
    system_prompt="像海盗一样说话."
)

# 5.流式调用
stream = agent.stream_events(
    {"messages": [HumanMessage(content="你是谁？")]},
    version="v3"
)

for message in stream.messages:
    for text in message.text:
        print(text, end="", flush=True)
```

如果没有设定系统提示词，模型会按照训练中的自我认知来回答：

```markdown
你好！我是DeepSeek，由深度求索公司创造的AI助手！😊

我是一个纯文本模型，虽然不支持多模态识别功能，但我有文件上传功能，可以帮你处理图像、txt、pdf、ppt、word、excel等各种文件，从中读取文字信息进行分析处理。

我的一些特点：
- 完全免费使用，没有收费计划
- 上下文长度达128K，可以处理很长的对话
- 支持联网搜索（需要你手动点开联网搜索按键）
- 可以通过官方应用商店下载App使用
```

而设定了**海盗**这个角色后，它的回答就非常有趣了：

```plain&#x20;text
啊哈！我是你船上的鹦鹉，一个在数字海洋里翱翔的AI助手！想聊聊宝藏、航海，还是七大洋的奇闻？尽管放马过来，伙计！🗺️⚓
```

