# 6、langchain——Chain

## 详细版

在之前的学习中，我们接触了聊天模型、用于构建输入的聊天提示模板以及帮助解析输出的输出解析器。研究后可以发现，各类聊天提示模板、chat Openai 等聊天模型，还有 Comma separated list output Parser、identical output Parser 等输出解析器，都会调用 invoke 方法，这并非巧合，而是专门的设计，这些组件均实现了 Lang chain 的 Runnable 接口，因此都具备 invoke 方法。

```python
ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions}"),
    ("human", "列出5个{subject}色系的十六进制颜色码。")
]).invoke(...)

ChatOpenAI(model="gpt-3.5-turbo").invoke(...)

CommaSeparatedListOutputParser().invoke(...)
```

invoke 方法是 Lang chain 表达式语言里 Runnable 的通用调用方法，不同组件的 invoke 方法，其输入与输出有着明确的对应关系。对于 prompt template 提示模板，invoke 接收表示输入变量值的字典，返回 prompt value 提示值；对于 chatmodel 聊天模型，invoke 接收提示值或是消息列表，返回聊天信息；对于 outputparser 输出解析器，invoke 接收聊天信息，返回解析后的结果。由此可见，提示模板 invoke 方法的输出，是聊天模型 invoke 方法的输入，而聊天模型 invoke 方法的输出，又成为输出解析器 invoke 方法的输入，基于这样的衔接关系，我们可以一次性调用多次 invoke，直接得到最终结果，当然在实际使用中，提示模板或输出解析器并非都是必要的，上述只是完整的使用示范。

```python
output_parser.invoke(
    model.invoke(
        prompt.invoke(
            {"subject": "莫兰迪", "parser_instructions": parser_instructions}
        )
    )
)
```

除了层层调用 invoke 方法的方式，Lang chain 还支持另一种写法，即通过小竖杠形式的管道操作符，将前面组件的输出作为后面组件的输入。比如 prompt | model | output Parser 的写法，就代表把提示值传给模型，再把模型的输出传给解析器，这种写法被称为 Lang chain 表达式语言，简称为 LCPL。

```python
(prompt | model | output_parser).invoke(
    {
        "subject": "莫兰迪",
        "parser_instructions": parser_instructions
    }
)
```

我们将多个组件的一系列调用叫做 chain 链，这个命名十分直观，想要得到链运行的结果，调用的方法名同样是 invoke，便于记忆。调用链的 invoke 方法时，传入的参数为给第一个组件的参数，原因在于每个组件会把上一个组件的输出作为自身输入，整个链路中只有第一个组件缺少输入。

链的组合并非局限于上述的固定形式，具备很强的灵活性，比如中间的 chat model 可以替换成 LLM，提示模板和输出解析器也并非是链路中必须存在的组件。总而言之，借助链能够组合出复杂的操作流程，而通过 Lang chain 表达式语言，组件之间的上下游关系也能被清晰明了地展现出来。

## 易懂版

# 一、先认识三个核心 “角色”

在和大模型交互时，我们通常会用到三个关键 “工具”，它们各司其职：

提问模板（Prompt Template）

作用：提前写好一个 “提问框架”，里面留好空位（比如{subject}），等你填具体内容（比如 “莫兰迪”），就能生成完整的提问。

```python
# 创建一个提问模板
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions}"),  # 系统提示：告诉AI怎么整理结果
    ("human", "列出5个{subject}色系的十六进制颜色码。")  # 人类提问：留了{subject}空位
])
```

AI 大脑（Model）

作用：接收完整的提问，生成回答。这里使用豆包 OpenAI 兼容接口的大模型，配置专属参数。

```python
import os
from langchain_openai import ChatOpenAI

# 初始化豆包兼容接口的大模型
chat_model = ChatOpenAI(
    # 核心配置：豆包的OpenAI兼容接口地址（固定）
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # API密钥：从环境变量读取（替换为你的豆包API Key）
    api_key=os.getenv("ARK_API_KEY"),
    model="ep-m-20260212020325-2bpp5",
    temperature=1.2,  # 随机性，0-2之间，值越高回复越多样
    max_tokens=300,  # 回复最大token数
    frequency_penalty=1.5,  # 频率惩罚，减少重复内容
    top_p=0.8  # 核采样，控制回复多样性
)

```

结果整理器（Output Parser）

作用：把 AI 返回的 "原始回答"，整理成我们需要的格式（比如逗号分隔的列表、JSON 等）。

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()  # 把结果整理成逗号分隔的列表
```

# 二、传统方式：像 “俄罗斯套娃” 一样嵌套调用

在没有链式调用之前，我们需要一层一层地调用这三个工具，代码看起来像 "套娃"：

```python
# 1. 先让提问模板生成完整提问
filled_prompt = prompt.invoke({
    "subject": "莫兰迪",
    "parser_instructions": output_parser.get_format_instructions()  # 告诉AI结果格式
})

# 2. 再让AI根据提问生成回答
ai_response = model.invoke(filled_prompt)

# 3. 最后让结果整理器把回答整理成列表
final_result = output_parser.invoke(ai_response)
```

这种方式虽然能工作，但代码写起来繁琐，逻辑也不直观，像在一层一层 “拆包裹”。

# 三、新式方式：用 "流水线"（管道符|）串起来

LangChain 提供了一个更优雅的方式：用管道符|把三个工具串成一条 "流水线"，数据从左到右自动流转，一步到位：

```python
# 把提问模板、AI大脑、结果整理器串成一条流水线
chain = prompt | model | output_parser

# 把需要填的内容放进流水线，自动生成整理好的结果
final_result = chain.invoke({
    "subject": "莫兰迪",
    "parser_instructions": output_parser.get_format_instructions()
})
```

### 这个流水线是怎么工作的？

最左边的提问模板先接收你传入的{"subject": "莫兰迪", ...}，生成完整的提问。

然后把完整提问传给中间的AI 大脑，AI 生成回答。

最后把 AI 的回答传给最右边的结果整理器，整理成你需要的格式（比如列表）。

# 四、为什么要用这种 “流水线” 方式？

代码更简洁：不用写三层嵌套，一行代码就能完成从提问到整理结果的全流程。

逻辑更清晰：从左到右的顺序，就是数据处理的顺序，一眼就能看懂 “先做什么，再做什么”。

扩展性更强：如果以后要加新的工具（比如再加一个结果过滤器），只需要在流水线后面加| new_tool就行，不用改之前的代码。

# 五、完整可运行的例子

把上面的内容整合起来，就是一个完整的、可直接运行的代码示例：

```python

import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser

# 1. 创建三个核心工具

# 提问模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions}"),
    ("human", "列出5个{subject}色系的十六进制颜色码。")
])

# 豆包OpenAI兼容接口大模型
chat_model = ChatOpenAI(
    # 核心配置：豆包的OpenAI兼容接口地址（固定）
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # API密钥：从环境变量读取（替换为你的豆包API Key）
    api_key=os.getenv("ARK_API_KEY"),
    model="ep-m-20260212020325-2bpp5",
    temperature=1.2,  # 随机性，0-2之间，值越高回复越多样
    max_tokens=300,  # 回复最大token数
    frequency_penalty=1.5,  # 频率惩罚，减少重复内容
    top_p=0.8  # 核采样，控制回复多样性
)

# 结果整理器
output_parser = CommaSeparatedListOutputParser()

# 2. 串成流水线
chain = prompt | chat_model | output_parser

# 3. 传入参数，运行流水线
result = chain.invoke({
    "subject": "莫兰迪",
    "parser_instructions": output_parser.get_format_instructions()
})

print(result)  # 输出整理好的颜色码列表，比如 ['#B8A9A9', '#C9B7B7', ...]