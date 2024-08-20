import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
 
# model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)
 
article_text = """
总结： <extra_id_0> - 首页 > 风力发电 / 问答 文章 字 : 大 中 小 说 话 ( ) <extra_id_34> . 欢迎访问 您的当前位置: www.russ.com 相 关新闻 > 您好, Russ! 热度 关注 微信公众号  <extra_id_51> 你好。 我想问您这是否与我的专业相关? 我请随时提问,请直接告诉我,Russ。 您是rus!您好! Russ, 我是rus。 ... 更多 信息 显示 来自 中国电力行业,数据处理,模型预测或其他电力相关领域的问题。
user:我是谁
ans: 您好，根据您提供的信息，您似乎是参与了一个关于风力发电功率预测的项目团队成员。您的名字没有直接给出，但您可能是团队中的曲嘉骏、董启翰、王乐天、李振宇或李东朔之一，或者是与这个项目相关的其他人。此项目关注于风力发电功率的预测，采用了多种数据分析和人工智能
技术，包括数据预处理、数据分解、多元非线性回归、ARIMA-X、LSTM等模型来提升预测精度。如果您需要关于风力发电或者该项目具体技术细节的解答，请随时提问。
"""
 
input_ids = tokenizer(
    article_text,
    return_tensors="pt",
    truncation=True,
    max_length=2048
)
generated_tokens = model.generate(
    input_ids["input_ids"],
    attention_mask=input_ids["attention_mask"],
    max_length=512,
    min_length = 32,
    no_repeat_ngram_size=2,
    num_beams=4
)
summary = tokenizer.decode(
    generated_tokens[0],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print(summary)