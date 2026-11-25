from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    """查询天气需要的参数。"""

    location: str = Field(description="城市名或经纬度")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位，只能是 celsius 或 fahrenheit",
    )
    include_forecast: bool = Field(default=False, description="是否返回五天预报")


@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """查询天气。"""
    temp = 22 if units == "celsius" else 72
    result = f"{location} 当前温度：{temp} {units}"
    if include_forecast:
        result += "\n未来五天：晴"
    return result


print("工具名：", get_weather.name)
print("用途：", get_weather.description)
print("参数：", get_weather.args)
print("合法参数调用：", get_weather.invoke({"location": "杭州", "units": "celsius"}))

try:
    get_weather.invoke({"location": "杭州", "units": "kelvin"})
except Exception as exc:
    print("非法参数被拦下：", type(exc).__name__)
