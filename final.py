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
from transformers import pipeline,AutoTokenizer,AutoModelForSeq2SeqLM

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
    chunk_size=1000,                  # 指定每个文本块的目标大小
    chunk_overlap=0,                # 指定文本块之间的重叠字符数
    length_function=len,             # 用于测量文本长度的函数
    is_separator_regex=False,        # 指定`separators`中的分隔符是否应被视为正则表达式，这里设置为False，表示分隔符是字面字符
    separators=['。','，','\n\n', '\n', ',', '.',]
)

# embedding
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"   # embedding的模型，可调
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}   # 是否对生成向量归一化
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 加载PDF文档并分割
pdf_path = 'D:\EEAgent1\doc\大创结题报告-final.pdf'
pages = load_pdf(pdf_path)
# texts = pages
texts = text_splitter.split_documents(pages)

# 存储嵌入向量库
db = Chroma.from_documents(
    documents=texts,
    embedding=hf,
    persist_directory='vector_base/(1000,10,8.15)/chroma_db'
)
# db.persist()  # 立即持久化数据库

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
llm = Tongyi(model_name='qwen-max', temperature=1)  # 基础模型可以改，temperature改变模型回答特点
chain = chat_prompt | llm

messages = []
summary_counter = 0
# MAX_messages = 5

# 初始化摘要模型
tokenizer = AutoTokenizer.from_pretrained("t5-small")
summarize_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
summarizer = pipeline("summarization", model=summarize_model, tokenizer=tokenizer)

print("你好，我是电力行业专家，有任何关于电力的专业问题都可以向我提问！")
while True:

    # 定义问题
    question = input("user:")
    messages.append({'role': Role.USER, 'content': question})
# # 控制消息列表长度
#     if len(messages) > MAX_messages:
#         messages = messages[-MAX_messages:]  # 保留最后MAX_MESSAGES_HISTORY条消息

    # 相似度方法通过查询文本检索数据
    similarDocs = db.similarity_search(question, k=10)
    if not similarDocs:
        print("没有找到相关文档。")
    else:
        # print("找到相关文档。")
        summary_prompt = "".join([doc.page_content for doc in similarDocs])

        send_message = f"你可以参考一下信息：({summary_prompt})回答({question})这个问题，如果没有找到与({question})相关的信息请如实回答，不要编造"
        messages.append({'role': Role.USER, 'content': send_message})

        # 调用 chain.invoke
        ans = chain.invoke({"question": messages})
        print("ans:", ans)

        # 将模型的回答添加到消息列表中
        messages.append({'role': 'assistant', 'content': ans})

    # 如果messages达到一定长度，则进行摘要
    if len(messages) >= 2 and summary_counter == 0:
        history_text = "\n".join([msg['content'] for msg in messages])
        summary = summarizer(history_text, max_length=500, min_length=100, do_sample=False)
        messages = [{'role': Role.SYSTEM, 'content': summary[0]['summary_text']}]
        summary_counter += 1  # 更新摘要次数

    # 重置summary_counter，以便下次达到条件时再次进行摘要
    if summary_counter > 0:
        summary_counter = 0

    print(summary)