"""第 10 篇的小型教学辅助：资料、模型初始化和输出观察，不代替 Agent 循环。"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool


REQUIREMENTS = """[虚构教学资料：需求 REQ-010]
为管理后台编写登录功能实施方案，目标是手机号验证码登录。
密码登录暂缓，本次方案不要安排实现密码登录。教学原型使用 SQLite。
本次只完成资料分析和方案编写，不开发页面、不调用短信、不部署。
请继续读取补充约束资料，以免遗漏验证码规则。
"""

CONSTRAINTS = """[虚构教学资料：本章新增约束 CON-010]
验证码发送须有冷却时间，并分别按手机号和 IP 限流；具体阈值待业务确认。
验证码必须设置有效期且成功校验后失效，错误次数达到上限后拒绝继续尝试；数值待确认。
方案应处理发送失败、验证码错误、过期和尝试超限，不能把短信发送成功当作登录成功。
登录成功后建立会话；这里只描述方案，不实现鉴权系统。
SQLite 是教学原型存储选择，生产扩展、并发和多实例限流需要另外评估。
"""

SYSTEM_PROMPT = """你是登录功能方案助手。使用中文，资料均为虚构教学设定。
你本次完成的是分析和编写实施方案，绝不是实现或部署登录功能。
请使用 write_todos 跟踪本次实际分析工作：读取需求、核对约束、编写并核对方案等。
不要把未来开发事项当作你已经执行的任务。未来开发步骤应写在最终实施方案中。
先建立待办，再调用 read_requirements，读完后单独调用 read_constraints。
两份资料不要并行读取，以便观察发现新信息的过程。
发现补充约束带来新工作时，显式补充或细化未完成待办，不要只修改状态。
用户追加要求时建立后续工作待办，保留相关上下文，及时更新进度。
每次 write_todos 提交完整列表；最后一次更新之后，再发一条包含实际成果的最终回答。
最终方案包含范围、实施步骤、异常处理和待确认事项，简洁但保留关键约束。
不要自行编造尚未确认的阈值；不要调用资料读取与 write_todos 以外的工具。
"""

REQUEST = "请读取登录需求和补充约束，完成登录功能实施方案。请展示本次分析工作的待办进度。"
FOLLOWUP = "请在刚才的方案中增加验证码登录验收清单，覆盖正常登录、发送限流、错误、过期及尝试超限；保留已有范围与待确认事项。"


def initialize():
    if not Path('../LangChain-任务规划.md').is_file():
        raise RuntimeError('请将 Jupyter 内核工作目录设置为本章 notebooks/。')
    load_dotenv('../../../.env')


def live_model():
    if not os.getenv('DEEPSEEK_API_KEY'):
        raise RuntimeError('真实实验需要项目 .env 中的 DEEPSEEK_API_KEY；不会改用模拟模型。')
    return init_chat_model(
        'deepseek-v4-flash', model_provider='deepseek',
        extra_body={'thinking': {'type': 'disabled'}},
        max_tokens=2600, timeout=60, max_retries=0,
    )


@tool
def read_requirements() -> str:
    """读取本章固定的虚构登录需求，不访问外部系统。"""
    return REQUIREMENTS


@tool
def read_constraints() -> str:
    """读取本章固定的虚构验证码约束，需求阅读之后调用。"""
    return CONSTRAINTS


TOOLS = [read_requirements, read_constraints]


def observe_update(update, trace):
    """打印并保存真实 updates，trace 只用于观察，不写回 Agent State。"""
    for node, patch in update.items():
        if not isinstance(patch, dict):
            continue
        if 'todos' in patch:
            trace['todos'] = patch['todos']
            trace['snapshots'].append(patch['todos'])
            print(f'[{node}] todos:', json.dumps(patch['todos'], ensure_ascii=False))
        messages = patch.get('messages', [])
        if isinstance(messages, (AIMessage, ToolMessage)):
            messages = [messages]
        for message in messages:
            trace['messages'].append(message)
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    print(f'[{node}] 调用 {call["name"]}:', json.dumps(call['args'], ensure_ascii=False))
                if message.content:
                    print(f'[{node}] 助手:', message.content)
            elif isinstance(message, ToolMessage):
                print(f'[{node}] 工具 {message.name}, status={message.status}:', message.content)


def final_answer(trace):
    for message in reversed(trace['messages']):
        if isinstance(message, AIMessage) and not message.tool_calls and message.content:
            return message.content
    return ''
