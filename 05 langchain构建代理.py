import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import chat_agent_executor

# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# 创建模型
model = ChatOpenAI(base_url="https://api.deepseek.com",
                   model="deepseek-v4-flash",
                   api_key=os.getenv("DEEPSEEK_API_KEY", ""))

# 没有任何代理的情况下
# print(model.invoke([HumanMessage(content='武汉最近的天气如何')]))

# LangChain 内置了一个工具，可以轻松地使用Tavily搜索引擎作为工具
search = TavilySearchResults(max_results=2)  # max_results=2 : 只返回两个结果
# print(search.invoke('武汉最近的天气怎么样'))


# 让模型绑定工具
# model_with_tool = model.bind_tools([search])

# 模型可以自动推理：是否需要调用工具去回答用户的提问
# resp1 = model_with_tool.invoke([HumanMessage(content='中国首都是什么')])
# print(f'model_result:{resp1.content}')   # model_result:中国的首都是 **北京**。
# print(f'tool_result:{resp1.tool_calls}')   # tool_result:[]
# resp2 = model_with_tool.invoke([HumanMessage(content='武汉最近的天气如何')])
# 
# print(f'model_result:{resp2.content}')   # model_result:
# print(f'tool_result:{resp2.tool_calls}') # tool_result:[{'name': 'tavily_search_results_json', 'args': {'query': '武汉 天气预报 2025'}, 'id': 'call_00_l8Px2g1TnTl70MSV13oe6609', 'type': 'tool_call'}]


# 创建代理
tools = [search]
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)
resp1 = agent_executor.invoke({'messages': [HumanMessage(content='中国首都是什么?')]})
print(resp1['messages'])

resp2 = agent_executor.invoke({'messages': [HumanMessage(content='武汉最近的天气如何?')]})
print(resp2['messages'])
# 查看调用工具之后的回答
print(resp2['messages'][2].content)
