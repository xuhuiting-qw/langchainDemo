import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pandas.io.sql import SQLDatabase

# 加载 .env 文件中的环境变量
load_dotenv()

# langsmith 的监控
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
# os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# 创建模型
model = ChatOpenAI(base_url="https://api.deepseek.com",
                   model="deepseek-v4-flash",
                   api_key=os.getenv("DEEPSEEK_API_KEY", ""))


# sqlalchemy 初始化MySQL数据库的连接
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'test_db8'
USERNAME = 'root'
PASSWORD = '123123'
# mysqlclient驱动URL
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)

# 测试连接是否成功
# print(db.get_usable_table_names())
# print(db.run('select * from t_emp limit 10;'))

# 直接使用大模型和数据库整合,只能根据你的问题生成sql语句
# 初始化生成sql语句的chain
test_chain = create_sql_query_chain(model, db)
# resp = test_chain.invoke({'question':'请问员工表中有多少条记录'})
# print(resp)


answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答: """
)

# 创建一个执行sql语句的工具
execute_sql = QuerySQLDatabaseTool(db=db)

# 1、生成sql 2、执行sql
# 3、模板
chain = (RunnablePassthrough.assign(query=test_chain).assign(result=itemgetter('query') | execute_sql)
        | answer_prompt
        | model
        | StrOutputParser()
         )

resp = chain.invoke({'question':'请问员工表中有多少条记录'})
print(resp)











