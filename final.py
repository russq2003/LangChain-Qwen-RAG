import os
import time
import warnings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredFileLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Tongyi
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from dashscope.api_entities.dashscope_response import Role

# 开始计时
start_time = time.time()
warnings.filterwarnings("ignore")

# 文档读取
def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

# 文档切片
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,                  # 指定每个文本块的目标大小
    chunk_overlap=50,                # 指定文本块之间的重叠字符数
    length_function=len,             # 用于测量文本长度的函数
    is_separator_regex=False,        # 指定`separators`中的分隔符是否应被视为正则表达式，这里设置为False，表示分隔符是字面字符
    separators=['\n\n', '\n', ',', '.', '。','，']
)

# embedding
model_name = "sentence-transformers/all-mpnet-base-v2"   # embedding的模型，可调
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}   # 是否对生成向量归一化
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 加载PDF文档并分割
pdf_path = 'D:\EEAgent1\doc\大创结题报告-final.pdf'
pages = load_pdf(pdf_path)
texts = text_splitter.split_documents(pages)

# 存储嵌入向量库
db = Chroma.from_documents(
    documents=texts,
    embedding=hf,
    persist_directory='WINDPOWER-FINAL/chroma_db'
)
db.persist()  # 立即持久化数据库

# Qwen模型API
DASHSCOPE_API_KEY = 'sk-a2da8256eb84494bb91e1f27488a65cb'
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 定义系统消息
system_message = SystemMessagePromptTemplate.from_template("你是一个电力行业专家，请回答用户问题，如果遇到与电力不相关的问题请回答该问题与电力不相关。对于不知道的问题就请回答不知道。")

# 定义人类消息
human_template = "{question}"
human_message = HumanMessagePromptTemplate.from_template(human_template)

# 创建聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# llm
llm = Tongyi(model_name='qwen-plus', temperature=0.1)  # 基础模型可以改
chain = chat_prompt | llm

messages = []
while True:
    # 定义问题
    question = input("user:")
    messages.append({'role': Role.USER, 'content': question})

    # 相似度方法通过查询文本检索数据
    similarDocs = db.similarity_search(question, k=10)
    if not similarDocs:
        print("没有找到相关文档。")
    else:
        # print("找到相关文档。")
        summary_prompt = "".join([doc.page_content for doc in similarDocs])

        send_message = f"下面的信息({summary_prompt})是否有这个问题({question})有关，如果你觉得无关请直接做出自己的回答，不要说与提供的信息无关；否则请根据{summary_prompt}对{question}的问题进行回答"
        messages.append({'role': Role.USER, 'content': send_message})

        # 调用 chain.invoke
        ans = chain.invoke({"question": messages})
        print("ans:", ans)

        # 将模型的回答添加到消息列表中
        messages.append({'role': 'assistant', 'content': ans})