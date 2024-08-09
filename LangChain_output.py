import os
from langchain_community.llms import Tongyi
from langchain.prompts.chat import (ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,)
from dashscope.api_entities.dashscope_response import Role

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
    whole_message = ''
    # 调用 chain.invoke
    ans = chain.invoke({"question": messages})

    print("ans:",ans)
    print()

    # 将模型的回答添加到消息列表中
    for response in ans:
        whole_message += response

    messages.append({'role':'assistant','content':whole_message})

# 解决流式输出记忆问题的思路：目标是要模型记住用户和自己说的话，同时只对用户的最后一个问题进行回答。
# 基于此，首先在循环外部创建一个新的列表messages，用来存储全部的信息。
# 在循环内部首先将用户提问输入messages。
# 在循环内部创建空列表whole_message，用来保存模型在循环内部的这个回答，同时将其作为assistant添加进messages。