import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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

class Classification(BaseModel):
    """
        定义一个Pydantic的数据模型，未来需要根据该类型，完成文本的分类
    """
    # 文本的情感倾向，预期为字符串类型
    # sentiment:str = Field(description='文本的情感')
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"], description="文本的情感")


    # 文本的攻击性，预期为1到5的整数
    # aggressiveness: int = Field(description="描述文本的攻击性，数字越大表示越攻击性")
    aggressiveness: int = Field(..., enum=[1, 2, 3, 4, 5], description="描述文本的攻击性，数字越大表示越攻击性")

    # 文本使用的语言，预期为字符串类型
    # language: str = Field( description="文本使用的语言")
    language: str = Field(..., enum=["spanish", "english", "french", "中文", "italian"], description="文本使用的语言")

# 创建一个用于提取信息的提示模板
tagging_prompt = ChatPromptTemplate.from_template(
    """
    从以下段落中提取所需信息。
    只提取'Classification'类中提到的属性.
    段落：
    {input}
    """
)

# 得到链
chain = tagging_prompt | model.with_structured_output(schema=Classification,method="function_calling")
input_text = "中国人民大学的王教授：师德败坏，做出的事情实在让我生气！"
resp = chain.invoke({'input':input_text})
print(resp)

input_text = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
resp = chain.invoke({'input':input_text})
print(resp)

