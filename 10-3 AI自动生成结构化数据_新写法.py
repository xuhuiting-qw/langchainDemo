import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

# 创建模型
model = ChatOpenAI(base_url="https://api.deepseek.com",
                   model="deepseek-chat",
                   api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                   temperature=0.8)

# 1、定义数据模型
class MedicalBilling(BaseModel):
    patient_id: int   # 患者ID，整数类型
    patient_name: str  # 患者姓名，字符串类型
    diagnosis_code: str  # 诊断代码，字符串类型
    procedure_code: str  # 程序代码，字符串类型
    total_charge: float  # 总费用，浮点数类型
    insurance_claim_amount: float  # 保险索赔金额，浮点数类型


# 2、构建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个医疗账单数据生成器。请严格按照以下格式生成数据：

参考样例：
Patient ID: 123456, Patient Name: 张娜, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350
Patient ID: 789012, Patient Name: 王兴鹏, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120
Patient ID: 345678, Patient Name: 刘晓辉, Diagnosis Code: E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250

要求：生成的数据必须与以上样例不重复。"""),
    ("human", "主题：{subject}\n额外要求：{extra}")
])

# 3、构建链
chain = prompt | model.with_structured_output(schema=MedicalBilling,method="function_calling")

# 4、批量生成（并行请求，比循环快）
inputs = [{
    "subject": "医疗账单",
    "extra": f"名字使用比较生僻的人名，如：昝、逄、郏、禚等姓氏"
} for _ in range(10)]

results = chain.batch(inputs)

for i, r in enumerate(results, 1):
    print(f"第{i}条: {r}")
