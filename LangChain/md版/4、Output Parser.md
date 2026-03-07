# 4、Output Parser

## 详细版

在前面我们学习了如何用提示模板构建出给AI模型的输入以及调用invoke方法，得到模型的回应。当我们在网页端和AI聊天，助手交互的时候得到回应，可能就表示这轮互动结束了，但是用代码交互时，通常会有对回应的后续操作。比如把AI的回答在我们自己的网页上展现，或者是提取回答里的一些信息等等。比如我们有个品牌网站，想更换不同网页的背景颜色。可以让AI每天生成五个符合要求的颜色，色号让那些色号自动更新背景，但相应的挑战是和代码逻辑生成内容的高确定性不太一样啊AI本质上是在根据概率生成内容。所以它的输出格式存在各种各样的可能性，这加大了我们从回复中提取信息的难度，而langchain的output parser输出解释器可以在这方面帮到我们。

输出解析器会做两方面事情:

第一，它能给模型下指令，要求模型按照指定的格式输出。

第二，它能解析模型的输出，帮助我们提取想要的信息。

在前面的例子里，我们想从AI的回答里提取五个颜色色号，那可以使用langchain的output parsers模块下的CommaSeparatedListOutputParser逗号分隔列表输出。

```python

from langchain.output_parsers import CommaSeparatedListOutputParser

```

也就是说，通过CommaSeparatedListOutputParser的指令，预期的模型输出会使用逗号分隔的元素所组成的字符串。比如长这样。这种输出格式很适合解析成拍送列表，所以CommaSeparatedListOutputParser也能帮我们把文本解析成列表。那具体要怎么做呢？首先我们还是可以先用chat promp的template消息提示模板，把给AI的消息列表写出来。

```python

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出5个{subject}色系的十六进制颜色码。")
    ]
)

```

那第一条消息需要是系统提示，后续跟上用户提示，那么输出解析器自带的指令，我们就会在之后填进系统消息里。要获得解析器的指令，得先有解析器才行，所以创建一个CommaSeparatedListOutputParser的实例，这个类有个叫get format instructions的方法。调用后会返回给我们这个逗号分隔列表输出解析器的文字指令，可以打印出来看看这个指令内容很直观。

```python

output_parser = CommaSeparatedListOutputParser()
parser_instructions = output_parser.get_format_instructions()

print(parser_instructions)
# 输出：Your response should be a list of comma separated values, eg: `foo, bar, baz`

```

说的是你的回应，应该是一串以逗号分隔的值，例如“foo，bar，buzz”，那我们就可以把这个指令文本连带着用户提示里的变量值一起传给提示模板的invoke方法。因为它们在提示模板里都是未知的，invoke之后，我们就得到了最终要传给模型的提示。

```python

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出5个{subject}色系的十六进制颜色码。")
    ]
)

final_prompt = prompt.invoke({
    "subject": "莫兰迪",
    "parser_instructions": parser_instructions
})

```

运行模型的invoke函数后可以看到啊，这个输出结果完全符合我们的要求，每个16进制颜色，色号之间都是用逗号分隔的。但我们也不需要自己解析文本，而是可以使用输出解析器的invoke方法去解析，具体解析过程应该和解析器前面给AI的指令相对应，因为CommaSeparatedListOutputParser给AI的指令里说过。回应应该是一串以逗号分割的值，所以它也相应知道要如何解析，那么我们给invoke方法直接传入AI的回应返回的结果就直接是python列表了，非常方便。

```python

# 导入模型所需的依赖包import osfrom langchain_openai import ChatOpenAIfrom langchain.output_parsers import CommaSeparatedListOutputParserfrom langchain_core.prompts import ChatPromptTemplate# 初始化你的专属ChatOpenAI模型（配置完全复用你的参数）chat_model = ChatOpenAI(    # 豆包OpenAI兼容接口地址（固定不变）    base_url="https://ark.cn-beijing.volces.com/api/v3",    # 你的豆包API密钥，也可直接写字符串（如api_key="你的密钥"），从环境变量读取更安全    api_key=os.getenv("ARK_API_KEY"),    model="ep-m-20260212020325-2bpp5",    temperature=1.2,    # 随机性0-2，值越高回复越多样    max_tokens=300,     # 模型回复的最大token数量    frequency_penalty=1.5,  # 频率惩罚，减少回复的重复内容    top_p=0.8           # 核采样，辅助控制回复多样性)prompt = ChatPromptTemplate.from_messages([    ("system", "{parser_instructions}"),    ("human", "列出5个{subject}色系的十六进制颜色码。")])output_parser = CommaSeparatedListOutputParser()parser_instructions = output_parser.get_format_instructions()final_prompt = prompt.invoke({"subject": "莫兰迪", "parser_instructions": parser_instructions})response = chat_model.invoke(final_prompt)print(response.content)color_list = output_parser.invoke(response)print(color_list)

```

## 易懂版

之前我们学过用提示模板给 AI 模型准备输入内容，再调用invoke方法获取 AI 的回复。在网页上和 AI 聊天时，拿到回复基本就结束了；但用代码和 AI 交互时，拿到回复后往往还要做后续操作 —— 比如把 AI 给的颜色码放到自己的网站上，或是从回复里精准提取需要的信息。

举个简单的例子：如果你有一个品牌网站，想每天更换不同页面的背景颜色，希望 AI 每天生成 5 个符合要求的颜色码，让这些颜色码自动更新网站背景。但这里有个问题：代码执行的结果是固定的，而 AI 是按概率生成内容的，它的输出格式可能乱七八糟（比如加多余文字、换行），这会让你很难提取想要的颜色码。而 LangChain 的output parser（输出解析器）就是专门解决这个问题的工具。

### 输出解析器的两大核心作用

给 AI 下达 “格式指令”：要求 AI 必须按照你指定的格式输出内容（比如只许用逗号分隔）

自动解析 AI 回复：帮你把 AI 的输出精准转换成能直接用的格式（比如 Python 列表）

下面以 “提取 5 个颜色码” 为例，介绍 LangChain 里的CommaSeparatedListOutputParser（逗号分隔列表输出解析器），它能让 AI 输出逗号分隔的字符串，还能直接把字符串转成 Python 列表。导入这个解析器的代码如下：

```python



```

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
```

### 第一步：编写 ChatPromptTemplate 消息提示模板

我们需要先准备发给 AI 的消息模板，包含两部分：

系统提示：用来放后续要加的 “格式指令”

用户提示：告诉 AI 具体要做什么（比如 “列出 5 个莫兰迪色系的颜色码”）

先把模板框架写好，格式指令后续再填充进去，代码如下：

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
```

### 第一步：编写 ChatPromptTemplate 消息提示模板

我们需要先准备发给 AI 的消息模板，包含两部分：

系统提示：用来放后续要加的 "格式指令"

用户提示：告诉 AI 具体要做什么（比如 "列出 5 个莫兰迪色系的颜色码"）

先把模板框架写好，格式指令后续再填充进去，代码如下：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions}"),
    ("human", "列出5个{subject}色系的十六进制颜色码。")
])
```

### 第二步：创建解析器实例并获取格式指令

要让 AI 按指定格式输出，首先得创建解析器的 “实例”（可以理解为 “激活这个工具”）。这个解析器有个get_format_instructions方法，调用后会生成一段文字指令，告诉 AI “该怎么输出”。代码和执行结果如下：

```python

output_parser = CommaSeparatedListOutputParser()
parser_instructions = output_parser.get_format_instructions()

print(parser_instructions)
# 输出：Your response should be a list of comma separated values, eg: `foo, bar, baz`

```

这段输出的意思很直白：AI 的回复必须是用逗号分隔的内容，比如 foo, bar, baz 这种形式，不能加多余的文字、换行。

### 第三步：渲染提示模板得到最终给 AI 的提示

“渲染提示模板” 其实就是把模板里的占位符（比如{subject}、{parser_instructions}）替换成实际内容。你把第二步拿到的格式指令，和想要的色系（比如 “莫兰迪”）传给模板的invoke方法，就能生成最终发给 AI 的完整提示。代码如下：

```python

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{parser_instructions}"),
        ("human", "列出5个{subject}色系的十六进制颜色码。")
    ]
)

final_prompt = prompt.invoke({
    "subject": "莫兰迪",
    "parser_instructions": parser_instructions
})

```

### 第四步：调用模型并解析输出结果

做好准备后，调用 AI 模型的invoke方法，AI 会严格按 “逗号分隔” 的格式输出颜色码。你不用手动拆分文字，直接用输出解析器的invoke方法，就能把 AI 的回复转换成 Python 列表。

下面是完整的可运行代码，包含 “初始化模型→准备模板→生成格式指令→渲染提示→调用 AI→解析结果” 的全流程，代码完全保留原样：

```python

# 导入模型所需的依赖包import osfrom langchain_openai import ChatOpenAIfrom langchain.output_parsers import CommaSeparatedListOutputParserfrom langchain_core.prompts import ChatPromptTemplate# 初始化你的专属ChatOpenAI模型（配置完全复用你的参数）chat_model = ChatOpenAI(    # 豆包OpenAI兼容接口地址（固定不变）    base_url="https://ark.cn-beijing.volces.com/api/v3",    # 你的豆包API密钥，也可直接写字符串（如api_key="你的密钥"），从环境变量读取更安全    api_key=os.getenv("ARK_API_KEY"),    model="ep-m-20260212020325-2bpp5",    temperature=1.2,    # 随机性0-2，值越高回复越多样    max_tokens=300,     # 模型回复的最大token数量    frequency_penalty=1.5,  # 频率惩罚，减少回复的重复内容    top_p=0.8           # 核采样，辅助控制回复多样性)prompt = ChatPromptTemplate.from_messages([    ("system", "{parser_instructions}"),    ("human", "列出5个{subject}色系的十六进制颜色码。")])output_parser = CommaSeparatedListOutputParser()parser_instructions = output_parser.get_format_instructions()final_prompt = prompt.invoke({"subject": "莫兰迪", "parser_instructions": parser_instructions})response = chat_model.invoke(final_prompt)print(response.content)color_list = output_parser.invoke(response)print(color_list)

```

运行代码后能看到两个结果：

print(response.content)：打印出 AI 按逗号分隔的颜色码（比如#E2D9C8, #D4C8B8, #C8B9A8, #BAA998, #ACA090）

color_list = output_parser.invoke(response)：直接返回 Python 列表（比如['#E2D9C8', '#D4C8B8', '#C8B9A8', '#BAA998', '#ACA090']），你可以直接用这个列表做后续的网站背景更新等操作。

### 总结

输出解析器的核心价值：既约束 AI 输出格式，又自动把 AI 回复转成可直接使用的 Python 格式（如列表），不用手动处理格式问题。

CommaSeparatedListOutputParser 使用步骤：创建解析器→获取格式指令→渲染提示模板→调用模型→解析输出。

完整代码可直接运行，运行后能同时得到逗号分隔的文本和可直接使用的 Python 列表，满足后续开发需求。