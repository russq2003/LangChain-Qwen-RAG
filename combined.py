import os
from langchain_community.llms import Tongyi
from langchain.prompts.chat import (ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,)
from dashscope.api_entities.dashscope_response import Role
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# 加载已有的向量数据库
model_name = "sentence-transformers/all-mpnet-base-v2"  # 要挂vpn...（下载到本地or换模型？）
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
db = Chroma(persist_directory="D:\EEAgent\project_2\chroma_db",embedding_function=hf)
print('success')

# Qwen模型API
DASHSCOPE_API_KEY = 'sk-a2da8256eb84494bb91e1f27488a65cb'
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 定义系统消息
system_message = SystemMessagePromptTemplate.from_template("你是一个电力行业专家，请回答用户问题，不知道就请回答不知道。")

# 定义人类消息
human_template = "{question}"
human_message = HumanMessagePromptTemplate.from_template(human_template)

# 创建聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# llm
llm = Tongyi(model_name='qwen-plus', temperature=0.1)
chain = chat_prompt | llm # 创建RunnableSequence

messages = []
while True:
    # 定义问题
    question = input("user:")
    messages.append({'role':Role.USER,'content':question})

    similarDocs = db.similarity_search(question,k=3) # 返回与查询question最相似的前5个文档
    print(similarDocs)
    summary_prompt = "".join([doc.page_content for doc in similarDocs])

    send_message = f"下面的信息({summary_prompt})是否有这个问题({question})有关，如果你觉得无关请告诉我无法根据提供的上下文回答'{question}'这个问题，简要回答即可，否则请根据{summary_prompt}对{question}的问题进行回答"
    messages.append({'role': Role.USER, 'content': send_message})

    whole_message = ''
    # 调用 chain.invoke
    ans = chain.invoke({"question": messages})

    print("ans:",ans)
    print()

    # 将模型的回答添加到消息列表中
    for response in ans:
        whole_message += response

    messages.append({'role':'assistant','content':whole_message})

