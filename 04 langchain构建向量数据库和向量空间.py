import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from openai.resources.beta.threads import messages

# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 聊天机器人案例
# 创建模型
model = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                   model='qwen3.7-plus')

# 准备测试数据，假设我们提供的文档数据如下：
documents = [
    # 每一个Document都代表一个文档数据
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",
        # 原数据
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

# 实例化一个向量空间
# chatgpt写法
# vector_store = Chroma.from_documents(documents,embedding=OpenAIEmbeddings())
# 使用阿里云的写法
vector_store = Chroma.from_documents(
    documents,
    embedding=DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v4",  # 支持 text-embedding-v3 / v4
    ),
)

# 做相似度的查询：返回的相似度的分数，分数越低相似度越高
# print(vector_store.similarity_search_with_score('咖啡猫'))

# 根据向量空间得到一个检索器
# bind(k=1)  返回相似度最高的第一个
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)

# print(retriever.batch(['咖啡猫', '鲨鱼']))

# 提示模板
message = """
使用提供的上下文仅回答这个问题。
{question}
上下文:
{context}
"""
prompt_temp = ChatPromptTemplate.from_messages([
    ('human', message)
])
# RunnablePassthrough()  允许我们将用户的问题之后再传递给prompt和model
chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | model

print(chain.invoke('请介绍一下猫').content)











