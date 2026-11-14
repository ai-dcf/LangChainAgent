# 5. 工具（Tools）

一个完整的Agent至少要包含两个关键的部分：

* **模型**：是Agent的大脑，负责推理、分析，规划任务步骤

* **工具**：是Agent的手脚，负责执行任务，与外界交互



所谓的**工具（Tool）**，本质就是一个可调用的**函数**，但是这个函数不是我们自己去调用，而是给模型调用。因此除了定义函数外，我们还需要清晰描述这个工具，让模型知道这个工具如何使用。包括下列信息：

* 工具名

* 工具的作用描述

* 工具需要的参数



在LangChain中，只要在工具函数上添加`@tool`装饰器，它就会自动提取函数信息，无需我们自己组装。不过`@tool`装饰器获取函数信息的方式有多种：

* 基于`@tool`装饰器属性

* 基于函数的文档注释

* 基于BaseModel作为schema



> **注意**：
>
> 在LangChain中，作为工具的函数**有两个保留的参数名**，你的自定义参数不能与之重复！他们是：
>
> * **config**：用来传递运行时配置
>
> * **runtime**：用来传递运行时上下文
>
> 
>
> 具体可参考官方文档：[保留参数名（Reserved argument names）](https://docs.langchain.com/oss/python/langchain/tools#reserved-argument-names)，后续我们会讲到这两个参数的用法。



## 5.1 基于装饰器属性描述工具

在LangChain中，定义工具需要用到@tool装饰器，我们可以通过装饰器来定义工具名、工具的作用：

```python
# 定义工具
from langchain.tools import tool

# =================方式一：基于装饰器描述工具==================
@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x ** 0.5
```



智能体在工作时，需要将函数的名称、输入、作用传递给大模型，默认情况下这些信息的来源是：

* 工具名称：函数名

* 工具输入：函数入参

* 工具作用：函数的注释





## 5.2 基于文档注释描述工具

如果不@tool装饰器没有定义工具名和作用描述，此时：

* 工具名：默认就是函数名

* 工具所需的参数：默认就是函数的参数列表

* 工具作用的描述：默认就是函数的文档注释（此时文档注释必须有，否则报错）

```python
# =================方式二：基于文档注释描述工具==================
from langchain.tools import tool

# 定义一个查询天气的tool
@tool
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """
    Get current weather and optional forecast.
    Args:
        location: city name or coordinates
        units: unit of degrees
        include_forecast: does it include the weather forecast
    """
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```



## 5.3 定义BaseModel描述工具参数

如果函数的参数比较多，而且比较复杂，此时建议通过pydantic model来描述参数列表。



例如，我们需要定义个查询天气的tool，借助于Pydantic来约束入参。

我们定义一个入参的模型，在模型中添加入参描述信息：

```python
# ==================方式三：基于BaseModel描述工具参数=================
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import tool
# 例如一个查询天气的tool
class WeatherInput(BaseModel):
    """查询天气的输入参数."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )
    
# 定义一个查询天气的tool
@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```



## 5.4 总结

虽然方式有三种，但大多数情况下都推荐使用第二种方式，即：tool装饰器 + 文档注释