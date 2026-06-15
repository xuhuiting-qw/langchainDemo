import os
from typing import Optional, List

from dotenv import load_dotenv
from langchain_classic import text_splitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.embeddings import DashScopeEmbeddings
import datetime
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic.v1 import BaseModel, Field
from langchain_core.documents import Document

# 加载 .env 文件中的环境变量
load_dotenv()
os.environ['http_proxy'] = '127.0.0.1:7897'
os.environ['https_proxy'] = '127.0.0.1:7897'
# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 创建模型
model = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                   model='qwen3.7-plus')

embeddings = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="text-embedding-v4",  # 支持 text-embedding-v3 / v4
)


# 指定目录
persist_dir = 'chroma_data_dir'  # 存放向量数据库的目录
# 一些YouTube的视频连接
urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]

# # document数组
# docs = []
# for url in urls:
#     # 一个youtube的视频对应一个document
#     docs.extend(YoutubeLoader.from_youtube_url(url,add_video_info=False).load())
#
# print(len(docs))
# print(docs[0])
# # 给docs添加额外的元数据：视频发布的年份
# for doc in docs:
#     doc.metadata['publish_year'] = int(
#             datetime.datetime.strptime(doc.metadata['publish_date'], '%Y-%m-%d %H:%M:%S').strftime('%Y'))
#
# print(docs[0].metadata)
# print(docs[0].page_content[:500])  # 第一个视频的字幕内容
#
# # 根据多个doc构建向量数据库
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=30)
# split_doc = text_splitter.split_documents(docs)
# # 向量数据库的持久化
# vectorstore =  Chroma.from_documents(split_doc,embeddings,persist_directory=persist_dir) # 并且把向量数据库持久化到磁盘

# 加载磁盘中的向量数据库
vectorstore =  Chroma(persist_directory=persist_dir,embedding_function=embeddings)

# 测试向量数据库的相似检索
# result = vectorstore.similarity_search_with_score('how do I build a RAG agent')
# print(result[0])
# print(result[0][0].metadata['publish_year'])

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# pydantic
# 构建数据模型
class Search(BaseModel):
    """
    定义了一个数据模型
    """
    # 内容的相似性和发布年份
    query:str = Field(None,description='Similarity search query applied to video transcripts.')
    publish_year:Optional[int] = Field(None,description='Year video was published')


chain = {'question':RunnablePassthrough()} | prompt | model.with_structured_output(Search)

# resp1 = chain.invoke('how do I build a RAG agent?')
# print(resp1)
# resp2 = chain.invoke('videos on RAG published in 2023')
# print(resp2)

def retrieval(search:Search)->List[Document]:
    _filter = None
    if search.publish_year:
        # 根据publish_year ，存在得到一个检索条件
        # $eq 是chroma向量数据库的固定语法
        _filter = {'publish_year':{'$eq'}}

    return vectorstore.similarity_search(search.query,filter=_filter)

new_chain = chain | retrieval

resp = new_chain.invoke('videos on RAG published in 2023')
print([(doc.metadata['title'],doc.metadata['publish_year'])  for doc in resp])














