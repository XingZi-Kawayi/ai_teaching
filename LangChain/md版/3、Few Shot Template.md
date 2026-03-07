# 3、Few Shot Template

小样本提示能让 AI 不用训练，快速适配新任务，核心是给 AI 传几个对话示例当参考，成本低还灵活。这些示例格式基本一致，只有具体内容不同，因此可以用模板批量构建。

LangChain 的prompts模块里，FewShotChatMessagePromptTemplate类就是专门干这个的，能高效搭建小样本示范，核心是传两个参数：example_prompt（示例模板）和examples（示例数据），下面一步步讲完整使用方法，第六点结合你的专属模型配置完善，可直接运行。

## 1. 定义示例模板

用ChatPromptTemplate做示例的对话模板，规定人机对话的固定格式，里面用{变量名}留空，后续填具体示例内容。注意：这个模板只用来做示范，不是最终给模型的用户提问。

```python

from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# 定义示范的对话格式，包含人机双方的消息
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "格式化以下客户信息：\n姓名 -> {customer_name}\n年龄 -> {customer_age}\n城市 -> {customer_city}"),
        ("ai", "## 客户信息\n- 客户姓名：{formatted_name}\n- 客户年龄：{formatted_age}\n- 客户所在地：{formatted_city}")
    ]
)

```

## 2. 准备示例数据

用字典列表存示例数据，列表里每个字典对应一个示范案例，字典的键要和上面模板里的{变量名}完全一致，值是具体的示范内容，给 AI 当参考。

```python
# 示例数据，键与模板变量一一对应
examples = [
    {
        "customer_name": "张三", "customer_age": "27", "customer_city": "长沙",
        "formatted_name": "张三", "formatted_age": "27岁", "formatted_city": "湖南省长沙市"
    },
    {
        "customer_name": "李四", "customer_age": "42", "customer_city": "广州",
        "formatted_name": "李四", "formatted_age": "42岁", "formatted_city": "广东省广州市"
    },
]
```

## 3. 构建小样本示范模板

把第一步的示例模板和第二步的示例数据，传给FewShotChatMessagePromptTemplate，组合成可直接使用的小样本示范模板。

```python
# 拼接模板和数据，生成小样本示范模板
few_shot_template = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
```

## 4. 拼接完整的模型输入模板

AI 的输入要求最后一条必须是用户提问，所以用ChatPromptTemplate.from_messages把「小样本示范模板」和「用户提问模板」拼起来，形成完整的模型输入模板。这个方法不仅能传人机对话元组，还能直接传做好的模板，灵活性很高。

```python
# 完整模板：示范在前，用户实际问题在后
final_prompt_template = ChatPromptTemplate.from_messages(
    [
        few_shot_template,  # 小样本示范内容
        ("human", "{input}")  # 用户提问位，{input}后续填实际问题
    ]
)
```

## 5. 填充内容，生成模型可识别的输入

模板里的{input}是未赋值的变量，不能直接传给模型，需要调用invoke方法，传入字典给变量赋值，生成模型能识别的聊天提示值。

```python
# 填充用户实际问题，生成最终输入
final_prompt = final_prompt_template.invoke({
    "input": "格式化以下客户信息：\n姓名 -> 王五\n年龄 -> 31\n城市 -> 郑州"
})
```

## 6. 初始化专属模型并传入输入运行（完善版）

这一步结合你提供的ChatOpenAI模型配置，先导入必要依赖，再初始化模型，最后将生成的消息列表传给模型，获取并打印结果，代码可直接复用（仅需替换自己的 API Key）。

```python
# 导入模型所需的依赖包
import os
from langchain_openai import ChatOpenAI

# 初始化你的专属ChatOpenAI模型（配置完全复用你的参数）
chat_model = ChatOpenAI(
    # 豆包OpenAI兼容接口地址（固定不变）
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 你的豆包API密钥，也可直接写字符串（如api_key="你的密钥"），从环境变量读取更安全
    api_key=os.getenv("ARK_API_KEY"),
    model="ep-m-20260212020325-2bpp5",
    temperature=1.2,    # 随机性0-2，值越高回复越多样
    max_tokens=300,     # 模型回复的最大token数量
    frequency_penalty=1.5,  # 频率惩罚，减少回复的重复内容
    top_p=0.8           # 核采样，辅助控制回复多样性
)

# 将final_prompt的messages属性（模型可识别的消息列表）传给模型
response = chat_model.invoke(final_prompt.messages)

# 打印模型的回复结果，获取格式化后的内容
print(response.content)
```

## 核心优势

高效省时间：加新示例不用重写完整对话，只往examples列表里加新字典即可，示例越多效率提升越明显；

格式强统一：AI 会严格遵循模板里的格式要求（如年龄加 “岁”、城市补省份），不用额外做格式校验；

低成本灵活：无需训练 / 微调模型，仅通过示例就能让 AI 适配新任务，修改模板即可快速调整 AI 的输出要求；

可复用性高：模型配置和模板可单独封装，后续换任务只需修改示例模板和数据，核心代码无需变动。