import os
from typing import Optional, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 加载 .env 文件中的环境变量
load_dotenv()

# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 创建模型  非思考模式
model = ChatOpenAI(base_url="https://api.deepseek.com",
                   model="deepseek-chat",
                   api_key=os.getenv("DEEPSEEK_API_KEY", ""))


# pydantic ：处理数据，验证数据，定义数据格式，虚拟化和反虚拟化，类型转化等等

# 定义一个数据模型   作用：可以从文本中提取结构化数据
class Person(BaseModel):
    """
    关于一个人的数据模型
    """
    name: Optional[str] = Field(None, description='人的名字')

    hair_color: Optional[str] = Field(
        default=None, description="如果知道的话，这个人的头发颜色"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="以米为单位测量的高度"
    )


class ManyPerson(BaseModel):
    """
    数据模型类：代表多个人
    """
    people:List[Person]

# 定义提示模板
# 定义自定义提示以提供指令和任何其他上下文。
# 2) 引入额外的参数以考虑上下文（例如，包括有关提取文本的文档的元数据。）
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的提取算法。只从未结构化文本中提取相关信息。如果你不知道要提取的属性的值，返回该属性的值为null。",
        ),
        # 请参阅有关如何使用参考记录消息历史的案例
        # MessagesPlaceholder('examples'),
        ("human", "{text}")
    ]
)

# with_structured_output :模型的输出是一个结构化的数据
text = '马路上走来一个女生，长长的黑头发披在肩上，大概1米7左右'
chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema=Person,method="function_calling")
resp = chain.invoke(text)
print(resp)


text = '马路上走来一个女生，长长的黑头发披在肩上，大概1米7左右。走在她旁边的是她的男朋友，叫：刘海；比她高10厘米'
chain = {'text': RunnablePassthrough()} | prompt | model.with_structured_output(schema=ManyPerson,method="function_calling")
resp = chain.invoke(text)
print(resp)
