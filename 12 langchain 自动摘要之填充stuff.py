import os

from dotenv import load_dotenv
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import result

# 加载 .env 文件中的环境变量
load_dotenv()

# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 创建模型
model = ChatOpenAI(base_url="https://api.deepseek.com",
                   model="deepseek-chat",
                   api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                   temperature=0)

# 加载我们的文档。我们将使用 WebBaseLoader 来加载博客文章：
loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load()  # 得到整篇文章


# TODO stuff 的第一种写法
chain = load_summarize_chain(model, chain_type='stuff')
result = chain.invoke(docs)
print(result['output_text'])

# TODO stuff 的第二种写法 常用写法
prompt_template = """针对下面的内容，写一个简洁的总结摘要:
"{text}"
简洁的总结摘要:"""
prompt = PromptTemplate.from_template(prompt_template)

llm_chain = LLMChain(llm=model,prompt=prompt)

stuff_chain = StuffDocumentsChain(llm_chain=llm_chain,document_variable_name='text')
result = stuff_chain.invoke(docs)
print(result['output_text'])







