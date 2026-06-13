import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# os.environ['http_proxy'] = '127.0.0.1:7897'
# os.environ['https_proxy'] = '127.0.0.1:7897'
# os.environ['OPENAI_API_KEY'] = 'sk-你的密钥'
# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 调用大模型
# model = ChatOpenAI(model='gpt-4-turbo')
# 1、创建模型
model = ChatOpenAI(base_url="https://api.deepseek.com",
                   model="deepseek-v4-flash",
                   api_key=os.getenv("DEEPSEEK_API_KEY", ""))

# 2、准备prompt
msg = [
    SystemMessage(content='请将以下的内容翻译成意大利语'),
    HumanMessage(content='你好，请问你要去哪里？')
]
# 原始写法
# result = model.invoke(msg)
# print(result)

# 简单的解析响应数据
# 3、创建返回的数据解析器
paser = StrOutputParser()
# print(paser.invoke(result))

# 4、得到链
chain = model | paser

# 5、直接使用chain来调用
print(chain.invoke(msg))





