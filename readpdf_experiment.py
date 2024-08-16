import pdfplumber
import os
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.llms import Tongyi
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.chains import LLMChain

# 文档读取
def load_pdf_with_pdfplumber(path):
    texts = []
    # tables = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return texts

# 文档切片
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
    separators=['。', '，', '\n\n', '\n', ',', '.']
)

# embedding
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 加载PDF文档并分割
pdf_path = 'D:\EEAgent1\doc\大创结题报告-final.pdf'
pages = load_pdf_with_pdfplumber(pdf_path)

# 将从pdfplumber获取的文本转换为Document对象
documents = [Document(page_content=page) for page in pages]

# 使用split_documents方法处理Document对象列表
texts = text_splitter.split_documents(documents)

# 存储嵌入向量库
db = Chroma.from_documents(
    documents=texts,
    embedding=hf,
    persist_directory='vector_base/(1000,0,pdfplumber)/chroma_db'
)

# 立即持久化数据库
# db.persist()

# Qwen模型API
DASHSCOPE_API_KEY = 'sk-a2da8256eb84494bb91e1f27488a65cb'
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 定义系统消息
system_message = SystemMessagePromptTemplate.from_template(
    "你是一个电力行业专家，请回答用户问题，如果遇到与电力不相关的问题请回答该问题与电力不相关。对于不知道的问题就请礼貌地回答不知道，尽最大可能回答用户问题。"
)

# 定义人类消息
human_template = "{question}"
human_message = HumanMessagePromptTemplate.from_template(human_template)

# 创建聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# llm
llm = Tongyi(model_name='qwen-max', temperature=1)

# 创建链式模型
chain = LLMChain(llm=llm, prompt=chat_prompt)

messages = []
MAX_messages = 5
print("你好，我是电力行业专家，有任何关于电力的专业问题都可以向我提问！")
while True:
    # 定义问题
    question = input("user:")
    messages.append({'role': 'user', 'content': question})
    
    # 控制消息列表长度
    if len(messages) > MAX_messages:
        messages = messages[-MAX_messages:]

    # 相似度方法通过查询文本检索数据
    similarDocs = db.similarity_search(question, k=10)
    if not similarDocs:
        print("没有找到相关文档。")
    else:
        summary_prompt = "".join([doc.page_content for doc in similarDocs])
        
        send_message = f"你可以参考一下信息：({summary_prompt})回答({question})这个问题，如果没有找到与({question})相关的信息请如实回答，不要编造"
        messages.append({'role': 'user', 'content': send_message})

        # 调用 chain.run，注意这里的参数应是问题本身
        ans = chain.run(messages)

        print("ans:", ans)

        # 将模型的回答添加到消息列表中
        messages.append({'role': 'assistant', 'content': ans})