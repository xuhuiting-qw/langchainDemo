import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory


# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 聊天机器人案例
# 创建模型
model = ChatOpenAI(base_url="https://api.deepseek.com",
                   model="deepseek-v4-flash",
                   api_key=os.getenv("DEEPSEEK_API_KEY", ""))

# 准备prompt
# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ('system','你是一个乐于助人的助手。用{language}尽你所能的回答所有问题'),
    MessagesPlaceholder(variable_name='my_msg')
])


# 得到链
chain = prompt_template | model

# 保存历史聊天记录
# 所有用户的聊天记录都保存到store。key：sessionId ,value:历史聊天记录
store = {}

# 此函数预期将接收一个session_id并返回一个信息历史记录对象
def get_session_history(session_id: str):
    # 先判断用户session_id 是否在store中
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg' # 每次聊天时候发送msg的key
)

# 给当前会话定义个session_id
config = {'configurable':{'session_id':'zs1243'}}

# 第一轮聊天
response = do_message.invoke(
    {
        'my_msg':[HumanMessage(content='你好啊！我是xu')],
        'language':'中文'
    },
    config=config
)
print(response.content)

# 第二轮聊天
response = do_message.invoke(
    {
        'my_msg':[HumanMessage(content='请问我的名字是什么？')],
        'language':'中文'
    },
    config=config
)
print(response.content)

# 第三轮聊天：返回的数据是流式的
config = {'configurable':{'session_id':'ls1243'}}
for resp in do_message.stream(
        {
            'my_msg': [HumanMessage(content='请你给我讲一个笑话，字数不要太长')],
            'language': '英文'
        },
        config=config
):
    # 每一次resp都是一个token
    print(resp.content,end='-')

print(store)
