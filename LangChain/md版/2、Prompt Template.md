# 2、Prompt Template

# 一、基础角色消息 Prompt 模板

### 导言

prompt 提示指的是用户给大语言模型的输入内容，当我们使用网页端和 ChatGPT 之类的聊天助手对话时，给 AI 的提示都需要一个个手动构建，但用代码就不一样了，我们可以在提示里插入变量，在使用结构化的提示的同时，根据需要动态调整里面的示例或数据。灵活性更高，效率也更高。LangChain 的 Prompt Template 提示模板就可以用于动态构建给模型的消息。

### 三类核心角色模板

langchain.prompts 里有 3 个模板，对应聊天的 3 类角色，都能用from_template创建，字符串中{}包起来的是变量，自动识别无需声明，后续动态填值即可。

具体来说，聊天模型的提示模板包括：

SystemMessagePromptTemplate：定义模型角色/规则（如翻译、总结）

HumanMessagePromptTemplate：定义用户的输入指令

AIMessagePromptTemplate：定义模型的历史回复（多轮对话用）

### 模板创建示例（系统提示模板）

比如这个系统提示的例子，里面input_language和output_language都是变量，那我们之后可以用同一个系统提示模板做中文翻译，法语，英语翻译，西班牙语都行。而且输入这个系统提示模板的input_variables属性，可以看到input_language和output_language被花括号包围，显然它们都自动被识别成了变量，不需要我们专门指出来。

```python

from langchain.prompts import SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate

system_template_text = "你是一位专业的翻译，能够将{input_language}翻译成{output_language}，并且输出文本会根据用户要求的任何语言风格进行调整。请只输出翻译后的文本，不要有任何其它内容。"

system_prompt_template = SystemMessagePromptTemplate.from_template(system_template_text)

```

那同样的创建一个用户提示模板，就用HumanMessagePromptTemplate的from_template，仍然是传入一个表示提示内容的字符串。如果字符串里有花括号包围的文本，就会作为后续可以动态填充的变量。

```python

human_template_text = "文本：{text}\n语言风格：{style}"
human_prompt_template = HumanMessagePromptTemplate.from_template(human_template_text)

```

有了提示模板后，要填入值也很简单，用format方法用参数传入变量的值即可。系统消息模板在format后会返回system message。人类消息模板在format后会返回human message，以此类推。

```python

system_prompt = system_prompt_template.format(input_language="英语", output_language="汉语")
human_prompt = human_prompt_template.format(text="I'm so hungry I could eat a horse", style="文言文")

# 验证变量是否正确填入
print("系统提示词内容：")
print(system_prompt.content)
print("\n用户提示词内容：")
print(human_prompt.content)

```

我们能从运行后的输出看到变量值都被填入，进之前还含有未知值的模板里面了，要得到聊天模型的回应，仍然需要传入消息列表。

那把不同角色消息模板通过format填入值后得到的消息放入列表里作为参数即可。虽然相比于直接把消息放入列表给invoke方法，用提示模板要多调用一次format，但也能给我们很大的灵活性。

```python

chat_model = ChatOpenAI(
    # 1. 核心配置：豆包的OpenAI兼容接口地址（固定）
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 2. API密钥：从环境变量读取（替换为你的豆包API Key）
    api_key=os.getenv("ARK_API_KEY"),
    model="ep-m-20260212020325-2bpp5",
    temperature=1.2,  # 随机性，0-2之间，值越高回复越多样
    max_tokens=300,     # 回复最大token数
    frequency_penalty=1.5,  # 频率惩罚，减少重复内容
    top_p=0.8           # 核采样，控制回复多样性
)

response = model.invoke([
    system_prompt,
    human_prompt
])

print(response.content)

```

### 批量处理场景应用

比如如果我们有一系列要翻译的文本，它们的源语言目标语言翻译出的风格要求都不一样。要相信提示模板的话，就不需要对提示一个一个硬编码了，接着for循环用同一个模板填入不同的值即可。我们能一次性得到不同语言，不同风格的多个翻译结果。

```json

input_variables = [
    {
        "input_language": "英语",
        "output_language": "汉语",
        "text": "I'm so hungry I could eat a horse",
        "style": "文言文"
    },
    {
        "input_language": "法语",
        "output_language": "英语",
        "text": "Je suis désolé pour ce que tu as fait",
        "style": "古英语"
    },
    {
        "input_language": "俄语",
        "output_language": "意大利语",
        "text": "Сегодня отличная погода",
        "style": "网络用语"
    },
    {
        "input_language": "韩语",
        "output_language": "日语",
        "text": "너 정말 짜증나",
        "style": "口语"
    }
]

for input_var in input_variables:
    response = model.invoke([
        system_prompt_template.format(
            input_language=input_var["input_language"],
            output_language=input_var["output_language"]
        ),
        human_prompt_template.format(
            text=input_var["text"],
            style=input_var["style"]
        )
    ])
    print(response.content)

```

# 二、一站式整合模板：ChatPromptTemplate

### 模板简介

除了使用上述三类角色分明的基础提示词模板，若想简化开发流程，可直接使用一站式整合模板ChatPromptTemplate。该类提供from_messages方法，从方法名可明确其支持接收一系列消息模板，无需单独创建不同角色的模板，大幅简化代码编写。

### 模板创建规则

ChatPromptTemplate.from_messages方法的参数为一个列表，列表内每个元素代表一条消息模板，且每个消息模板以元组形式定义：

元组第一个元素：字符串类型，代表消息角色，可选值为system、human、AI

元组第二个元素：字符串类型，代表消息的具体内容，可包含{}包裹的变量

### 变量赋值方式

为模板中的变量赋值时，调用模板的invoke方法即可，该方法接收一个字典作为参数，字典的键值对与模板中的变量一一对应，可一次性为所有角色消息文本中的变量统一赋值，无需逐个为不同角色模板单独赋值。

```python

import os

# 导入LangChain的OpenAI兼容Chat模型（核心）
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位专业的翻译，能够将{input_language}翻译成{output_language}，并且输出文本会根据用户要求的任何语言风格进行调整。请只输出翻译后的文本。"),
        ("human", "文本: ```{text}```\n语言风格: {style}"),
    ]
)

prompt_value = prompt_template.invoke({
    "input_language": "英语",
    "output_language": "汉语",
    "text": "I'm so hungry I could eat a horse",
    "style": "文言文"
})

```

invoke 调用后的结果是得到一个 ChatPromptValue 的实例，我们可以把 ChatPromptValue（聊天提示值）看成是未知变量值被填入之后的聊天提示模板。要得到模型的回应和之前也是一模一样，往 invoke 方法里传入消息列表就搞定了。

```python

chat_model = ChatOpenAI(
    # 1. 核心配置：豆包的OpenAI兼容接口地址（固定）
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 2. API密钥：从环境变量读取（替换为你的豆包API Key）
    api_key=os.getenv("ARK_API_KEY"),
    model="ep-m-20260212020325-2bpp5",
    temperature=1.2,    # 随机性，0-2之间，值越高回复越多样
    max_tokens=300,     # 回复最大token数
    frequency_penalty=1.5,  # 频率惩罚，减少重复内容
    top_p=0.8           # 核采样，控制回复多样性
)

response = model.invoke(prompt_value)

```