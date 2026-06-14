import os
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

# 加载 .env 文件中的环境变量
load_dotenv()

# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
# os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# 创建模型
model = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                   model='qwen3.7-plus')


# TODO 1、加载数据：一篇博客内容数据
# WebBaseLoader：从网页 URL 加载内容
# bs4：解析 HTML 页面
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'],
    bs_kwargs=dict(
        # 只提取 post-header 、 post-title 、 post-content 这三个 CSS 类的内容，过滤无关元素
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )
)
# 实际请求网页并解析，返回 Document 对象列表
docs = loader.load()

# TODO 2、大文本的切割
'''
- RecursiveCharacterTextSplitter — 递归式切分器，按换行符 → 句号 → 空格 → 字符的优先级切分
- chunk_size=1000 — 每段最多 1000 个字符
- chunk_overlap=200 — 相邻片段重叠 200 字符，防止语义断裂
- split_documents(docs) — 将每个 Document 切分成多个小 Document
'''
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splits = splitter.split_documents(docs)
# 测试
# text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"
# res = splitter.split_text(text)
# for s in res:
#     print(s,end='*****-\n')

# TODO 3、存储
# Chroma:向量数据库，存储文档向量
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=DashScopeEmbeddings(
                                        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
                                        model="text-embedding-v4",  # 支持 text-embedding-v3 / v4
                                    ))

# TODO 4、检索器
# 将向量数据库包装成一个检索器，默认根据 余弦相似度 检索与问题最相关的文档片段
retriever = vectorstore.as_retriever()

# TODO 5、整合
# 创建一个问题的模板
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer concise.\n

{context}
"""

# MessagesPlaceholder:占位符，运行时动态插入消息列表
'''
ChatPromptTemplate — 消息模板，包含：
- system 消息：角色指令
- MessagesPlaceholder("chat_history") ：对话历史（多轮聊天记录）
- human 消息：用户当前输入
'''
prompt = ChatPromptTemplate.from_messages([  # 提问和回答的 历史记录 模板
    ('system',system_prompt),
    MessagesPlaceholder("chat_history"),
    ('human','{input}')
])

# TODO 6、得到chain
# 可以做问答
# create_stuff_documents_chain ：将检索到的文档"填充"到 Prompt 中，再传给 LLM
chain1 = create_stuff_documents_chain(model, prompt)

'''
注意：
一般情况下，我们构建的链（chain）直接使用输入问答记录来关联上下文。但在此案例中，查询检索器也需要对话上下文才能被理解。

解决办法：
添加一个子链(chain)，它采用最新用户问题和聊天历史，并在它引用历史信息中的任何信息时重新表述问题。这可以被简单地认为是构建一个新的“历史感知”检索器。
这个子链的目的：让检索过程融入了对话的上下文。
'''


# 子链的提示模板
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

#  主要是帮助检索器来理解上下文
retriever_history_temp = ChatPromptTemplate.from_messages([
    ('system',contextualize_q_system_prompt),
    MessagesPlaceholder('chat_history'),
    ('human','{input}')
])
# 创建一个子链
# create_history_aware_retriever：创建"历史感知"检索器，根据聊天历史重写问题
history_chain = create_history_aware_retriever(
    model,retriever,retriever_history_temp
)

# 保存问答的历史记录
'''
- store — 字典，key 为 session_id ，value 为该会话的聊天历史
- get_session_history — 根据 session_id 获取/创建历史记录对象
- ChatMessageHistory — 在内存中存储消息列表
'''
store = {}
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 创建一个父链chain:把前两个链整合
'''
create_retrieval_chain(history_chain, chain1) — 将历史感知检索器 + 问答链组合：
1. 先用 history_chain 重写问题并检索文档
2. 再把文档传给 chain1 进行问答
'''
chain = create_retrieval_chain(history_chain,chain1)

'''
RunnableWithMessageHistory — 给整个链注入记忆能力：
- input_messages_key='input' — 输入字段名
- history_messages_key='chat_history' — 历史字段名（对应 Prompt 中的占位符）
- output_messages_key='answer' — 输出字段名
'''
result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

# 第一轮对话
resp1 = result_chain.invoke(
    {'input': 'What is Task Decomposition?'},
    config={'configurable': {'session_id': 'ls111'}}
)
print(resp1['answer'])

# 第二轮对话
resp2 = result_chain.invoke(
    {'input': 'What are common ways of doing it?'},
    config={'configurable': {'session_id': 'ls111'}}
)
print(resp2['answer'])


