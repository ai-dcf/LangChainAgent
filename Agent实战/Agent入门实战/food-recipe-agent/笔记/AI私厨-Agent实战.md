本章我们要利用前面所学的知识实现一个多模态智能体应用：AI私厨。



# 1. 需求分析

AI私厨是一个基于LangChain和多模态AI的食谱推荐应用。用户可以拍摄自家冰箱或厨房的食物照片，AI会自动识别图片中的食材，根据食材搜索相关食谱推荐给用户。



## 1.1 功能特性

* 📸 **图片识别** - 上传食材图片，自动识别其中的食材

* 🔍 **智能搜索** - 根据识别的食材搜索相关食谱

* 🍽️ **智能排序** - 按推荐度、难度、营养价值对食谱进行排序

* 💡 **创意建议** - 当找不到合适食谱时，提供创意搭配建议

* 💬 **对话交互** - 聊天式界面，支持图片上传 + 文本对话



## 1.2 预期效果

基本的聊天界面窗口，支持**图片上传**+**文本对话**：

![](images/image-12.png)

上传图片，根据图片识别食材：

![](images/image-13.png)

![](images/image-9.png)

![](images/image-11.png)



根据食材搜索食谱，并智能排序：

![](images/image-10.png)

## 1.3 实现思路分析

定义一个基础Agent，核心就是模型（Model）和工具（Tool），我们从这两点入手：

* **模型**：由于要实现多模态输入，所以必须用多模态模型，比如 deepseek-flash

* **工具**：需要联网搜索食谱，所以要用到web搜索工具，例如：deepseek_web_search

* **记忆**：短期记忆我们暂时使用Sqlite来实现



剩下就是提示词了，这里我给大家准备了一份测试过的提示词：

```python
system_prompt = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。
2.智能食谱检索：优先调用 web_search 工具，以“可用食材清单”为核心关键词，查找可行菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。

请严格按照流程，优先调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥。
"""
```



接下来，就是组合这些东西，开发你的智能体了，赶紧动手试试吧！









# 2. 实战开发

Agent跑通了，但目前还存在几个问题：

* 目前的图片信息还是采用base64方式提交给模型，会占用大量内存，性能差

* 我们没有开发自己的前端，用户体验不好



接下来，我们就逐一解决这些问题。



在向模型提交多模态消息，比如：音频、视频、图片时，我们不建议直接发送文件数据（base64）给模型，这会大量占用内存和会话记忆。更常见的方案是：

* 先将多模态文件上传至通用的OSS服务，例如：阿里云OSS、腾讯云COS等

* 获取oss服务的文件url地址，组织多模态消息，发送给大模型



因此，我们需要单独开发一个文件上传的服务接口，让前端先上传好文件，再调用Agent.

![](images/image-16.png)

我们的服务端需要具备以下接口：

* 对话接口：接收用户聊天消息，并调用Agent

* 会话管理接口：查询或删除会话历史

* 文件上传接口：调用OSS提供的客户端，实现文件上传授权，将来由前端完成文件上传，文件不经过服务器。



## 4.1 FastAPI服务端

首先，我们需要安装一些依赖，包括FastAPI和阿里云的OSS：

```powershell
uv add fastapi alibabacloud-oss-v2
```



在资料中已经提供好了一个基于FastAPI开发的基础网关：

![](images/image-17.png)

项目结构：

```bash

  app/
  ├── main.py                    # FastAPI 入口，配置路由和静态文件
  │
  ├── agents/
  │   └── personal_chief.py      # AI 代理核心逻辑
  │ 
  ├── api/
  │   └── v1/
  │       ├── chat.py             # 对话 API
  │       │   ├── POST /chat/stream     流式对话
  │       │   ├── GET  /chat/messages   获取历史
  │       │   └── DELETE /chat/messages 清空历史
  │       │
  │       └── oss.py              # OSS 上传签名 URL
  │
  ├── models/
  │   └── schemas.py              # Pydantic 数据模型，请求/响应数据结构定义  
  │
  ├── common/
  │   └── logger.py               # 日志配置
  │
  └── static/                     # Next.js 编译产出的静态网页 
      ├── index.html              # 前端入口
      ├── _next/                  # Next.js 构建资源
      └── ...                     # 其他静态资源
```



将所有内容复制到我们的app目录下：

![](images/image-18.png)

main.py作为程序入口：

```python
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import chat
from app.api.v1 import oss
from app.common.logger import setup_logging

# 初始化日志配置
setup_logging()

# 初始化FastAPI
app = FastAPI(
    title="Personal Chief API",
    description="私厨",
    version="0.1.0"
)

# ...中间代码略...

if __name__ == "__main__":
    import uvicorn
    # 启动命令：python -m app.main
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
```

直接运行main.py，访问：http://localhost:8001即可：

![](images/image-19.png)





## 4.2 OSS配置

虽然资料已经帮大家实现了oss的文件上传，但是大家需要自己去oss注册服务，并申请API\_KEY



这里我们以阿里云OSS为例来说明



### 4.2.1 注册阿里云

注册地址：

注册登录成功后，访问链接：https://oss.console.aliyun.com/overview，即可看到oss控制台：

![](images/image-20.png)



如果显示【**尚未开通**】，点击【**立即开通**】即可：

![](images/image-21.png)

默认为按量付费，价格非常便宜，几乎可以忽略不计：

![](images/image-22.png)

开通成功后，即可进入控制台页面：

![](images/image-23.png)





### 4.2.2 申请API\_KEY

访问链接：https://ram.console.aliyun.com/overview?activeTab=workflow，进入RAM访问控制页面：

![](images/image-24.png)

点击：`用户`>`创建用户`，填写用户信息：

![](images/image-25.png)

创建完成后，一定要记住你的AccessKey的ID和Secret：

![](images/image-26.png)



然后，给新添加的用户《新增授权》：

![](images/image-27.png)

在授权页面，给新用户添加oss的绝对控制权限：

![](images/image-28.png)



### 4.2.3 开通OSS

访问链接：https://oss.console.aliyun.com/overview，进入oss控制台，选择**创建Bucket**：

![](images/image-29.png)

填写bucket信息：

![](images/image-30.png)

### 4.2.4 设置权限

目前，新建的Bucket还是无法访问的，是私有的。为了方便测试，这里我们**暂时**将其设置为**公共读**。

> **注意**：
>
> * 实际开发中oss中的图片应该设置为**私有**！不可对外暴露！！由额外的CDN服务对外暴露！
>
> * 本例中，我们为了方便暂时将bucket设置为公共读，测试完毕后**请及时关闭权限**！



首先，进入Bucket管理页面：

![](images/image-31.png)

进入具体的Bucket设置，关闭公共访问开关：

![](images/image-32.png)

设置权限为公共读：

![](images/image-33.png)



接着，我们要开启跨域访问权限：

![](images/image-34.png)

点击《**创建规则**》，填写跨域规则：

![](images/image-35.png)

### 4.2.5 配置API\_KEY

最后，我们需要把阿里云OSS的API\_KEY配置到环境变量中，也就是项目的.env文件内，格式如下：

```properties
# 阿里OSS
OSS_ACCESS_KEY_ID=LTBI5tHzjC36KhCJfPqlbaCo
OSS_ACCESS_KEY_SECRET=aDPGBi1nIlYzcEmk5djWJGUv3w9Qkh
OSS_BUCKET=tmp9527
```





访问 http://localhost:8001，试试上传图片是否成功：

![](images/image-36.png)

选择图片，可以看到预览：

![](images/image-37.png)

点击发送按钮，上传成功应该能在聊天窗看到结果：

![](images/image-38.png)



## 4.3 添加Checkpointer

由于上一节是基于LangGraph和LangSmith的部署，所以Agent并没有添加Checkpointer，由LangGraph自己处理。而现在既然要自定义Agent部署， 就必须自己添加Checkpointer了。



我们依然使用Sqlite作为存储方案，修改personal\_chief.py文件，添加Checkpointer给Agent：

```python
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# 1.加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 2.web搜索工具，使用tavily作为web搜索工具
web_search = TavilySearch(
    max_results=5,
    topic="general"
)

# 3.多模态模型
model = init_chat_model(
    model="qwen3.5-plus",  # 模型名称，这里选择qwen3.5-plus，这是一个多模态模型，支持图片、文本、音频、视频
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 4.初始化checkpointer
# 连接sqlite
connection = sqlite3.connect("../db/personal_chief.db", check_same_thread=False)
# 初始化checkpointer
checkpointer = SqliteSaver(connection)
# 自动建表
checkpointer.setup()

# 5.Agent系统提示词
system_prompt = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。
2.智能食谱检索：优先调用 web_search 工具，以“可用食材清单”为核心关键词，查找可行菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。

请严格按照流程，优先调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥。
"""

# 6.创建Agent
agent = create_agent(
    model=model,  # 模型
    tools=[web_search],  # 工具
    checkpointer=checkpointer,  # 记忆
    system_prompt=system_prompt  # 系统提示词
)
```

在app下创建一个db目录，用以存放Sqlite的db文件：

![](images/image-39.png)



## 4.4 开发接口

目前在`agents/personal_cheif.py`文件中虽然已经开发了Agent，但是外界还是无法访问，我们需要定义Restful的接口供前端使用。



其中，在`api/v1/chat.py`中定义了Restful的接口，等待我们实现功能：

```python
from fastapi import APIRouter

router = APIRouter()

@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """流式聊天接口"""
    pass

@router.get("/chat/messages")
async def get_chat_messages(thread_id: str):
    """获取历史消息"""
    pass


@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str):
    """清空历史消息"""
    pass
```





### 4.4.1 功能实现

我们可以在`agents/personal_cheif.py`中基于agent开发所需的3个功能：

* 多模态聊天

* 获取会话历史

* 清空会话历史



`agents/personal_cheif.py`的完整代码如下：

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from app.common.logger import logger
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()

# web搜索工具，使用tavily作为web搜索工具
tavily = TavilySearch(
    max_results=5,
    topic="general"
)


# 多模态模型
model = init_chat_model(
    model="qwen3-omni-flash",  # 模型名称，这里选择qwen3.5-plus，这是一个多模态模型，支持图片、文本、音频、视频
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)


# 初始化checkpointer
checkpointer = SqliteSaver(sqlite3.connect("db/personal_chief.db", check_same_thread=False))
# 自动建表
checkpointer.setup()

# Agent系统提示词
system_prompt = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。
2.智能食谱检索：优先调用 web_search 工具，以“可用食材清单”为核心关键词，查找可行菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。

请严格按照流程，优先调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥。
"""

# 创建代理
agent = create_agent(
    model=model,  # 模型
    tools=[tavily],  # 工具
    checkpointer=checkpointer,  # 记忆
    system_prompt=system_prompt  # 系统提示词
)

# 流式对话
async def search_recipes(prompt: str, image: str, thread_id: str):
    """调用agent搜索食谱"""
    logger.info(f"[用户]: {prompt}, image: {image}, thread_id: {thread_id}")
    try:
        # 判断是否有图片，封装不同格式的消息
        if not image or image.strip() == "":
            message = HumanMessage(content=prompt)
        else:
            message = HumanMessage(content=[
                {"type": "image", "url": image},
                {"type": "text", "text": prompt}
            ])

        # 流式调用Agent
        for chunk, metadata in agent.stream(
            {"messages": [message]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(f"\n[错误]: {str(e)}")
        yield "信息检索失败，试试看手动输入食物列表？"

# 清空会话
def clear_messages(thread_id: str):
    """清空会话"""
    logger.info(f"清空历史消息，thread_id: {thread_id}")
    checkpointer.delete_thread(thread_id)

# 查询会话历史
def get_messages(thread_id: str) -> list[dict[str, str]]:
    """获取会话历史"""
    logger.info(f"获取历史消息，thread_id: {thread_id}")

    # 根据 thread_id 查询 checkpoint
    checkpoint = checkpointer.get({"configurable": {"thread_id": thread_id}})

    # 如果不存在，返回空列表
    if not checkpoint:
        return []

    # 安全获取 messages
    channel_values = checkpoint.get("channel_values")
    if not channel_values:
        return []

    messages = channel_values.get("messages", [])
    if not messages:
        return []

    # 转换消息格式
    result = []
    for msg in messages:
        if not msg.content:
            continue

        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})

    return result
```





### 4.4.2 完善接口

然后，修改chat.py文件，导入`agents.personal_cheif`中的三个方法，完善接口：

```python
from fastapi import APIRouter
from app.models.schemas import ChatRequest
from fastapi.responses import StreamingResponse
from app.agents.personal_chief import search_recipes, get_messages, clear_messages


router = APIRouter()


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """流式对话"""
    return StreamingResponse(
        search_recipes(request.message, request.image_url, request.thread_id),
        media_type="text/event-stream"
    )


@router.get("/chat/messages")
async def get_chat_messages(thread_id: str):
    """获取历史消息"""
    messages = get_messages(thread_id)
    return {"messages": messages}


@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str):
    """清空历史消息"""
    clear_messages(thread_id)
    return {"success": True}
```





## 4.5 测试

访问http://localhost:8001，即可实现多模态聊天了：

![](images/image-40.png)

