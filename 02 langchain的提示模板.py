import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

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
# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system','请将下面的内容翻译成{language}'),
    ('user','{text}')
])

# 简单的解析响应数据
# 3、创建返回的数据解析器
paser = StrOutputParser()
# print(paser.invoke(result))

# 4、得到链
chain = prompt_template | model | paser

# 5、直接使用chain来调用
# print(chain.invoke(msg))
print(chain.invoke({'language':'English','text':'我下午还有一节课不能去打球了'}))

# 把我们的程序部署成服务
# 创建fastAPI的应用
app = FastAPI(title='我的langchain服务',
              version='v1.0',
              description='使用langchain翻译任何语句的服务器')
# 添加路由
add_routes(
    app,
    chain,
    path='/chainDemo',
)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='localhost',port=8000)



