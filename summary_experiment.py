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
import torch

# # Qwen模型API
DASHSCOPE_API_KEY = 'sk-a2da8256eb84494bb91e1f27488a65cb'
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 指定存储向量数据库的目录
persist_directory = 'vector_base/(1000,10,8.19-1)/chroma_db'

# embedding
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"   # embedding的模型，可调
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}   # 是否对生成向量归一化
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# 加载Chroma数据库
db = Chroma(persist_directory=persist_directory,embedding_function=hf)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# model_checkpoint = "google/mt5-small"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

# # 生成摘要的函数
# def generate_summary(history_text):
#     input_ids = tokenizer(
#         history_text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512    # 限制最长输入
#     ).to(device)  # 将张量移动到GPU/CPU上

#     generated_tokens = model.generate(
#         input_ids["input_ids"],
#         attention_mask=input_ids["attention_mask"],
#         max_length=1024, # 限制最长总结输出
#         min_length = 128,
#         no_repeat_ngram_size=2,
#         num_beams=4
#     )

#     summary = tokenizer.decode(
#         generated_tokens[0],
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#     return summary

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

        send_message = f"你可以参考以下信息：({summary_prompt})回答({question})这个问题，如果没有找到与({question})相关的信息请如实回答，不要编造"
        messages.append({'role': Role.ATTACHMENT, 'content': send_message})

        # 调用 chain.invoke
        ans = chain.invoke({"question": messages})
        print("ans:", ans)

        # 将模型的回答添加到消息列表中
        messages.append({'role': 'assistant', 'content': ans})

    # 如果messages达到一定长度，则进行摘要
    if len(messages) >= 2 and summary_counter == 0:
        # history_text = "\n".join([msg['content'] for msg in messages if msg['role'] != 'attachment'])
        summary_message = f"对以上信息进行总结，着重提炼并记住在前面的信息"
        messages.append({'role':Role.BOT,'content':summary_message})
        # summary = generate_summary(history_text)
        filtered_messages = [msg for msg in messages if msg['role'] != 'attachment']
        summary = chain.invoke({"question":filtered_messages})
        messages = [{'role': Role.SYSTEM, 'content': summary}]
        summary_counter += 1  # 更新摘要次数

    # 重置summary_counter，以便下次达到条件时再次进行摘要
    if summary_counter > 0:
        summary_counter = 0

    print("")
    print("总结：",summary)