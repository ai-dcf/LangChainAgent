from langchain.tools import tool


@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x ** 0.5


print("工具名：", tool1.name)
print("用途：", tool1.description)
print("参数：", tool1.args)
print("直接调用：", tool1.invoke({"x": 467}))
