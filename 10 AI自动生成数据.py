import os

from dotenv import load_dotenv
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 创建模型
model = ChatOpenAI(base_url="https://api.deepseek.com",
                   model="deepseek-v4-flash",
                   api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                   temperature=0.8)

# 创建链
chain = create_data_generation_chain(model)

# 生成数据
result = chain(  # 给予一些关键词，随机生成一句话
    {
        'fields':['蓝色','黄色'],
        'preferences':{}
    }
)
print(result)

result = chain(  # 给予一些关键词，随机生成一句话
    {
        'fields':{'颜色':['蓝色','黄色']},
        'preferences':{'style':'让它想诗歌一样'}
    }
)
print(result)








