# 5、Output Parser——JSON

## 详细版

如何借助 Lang chain 的输出解析器让模型输出逗号分隔列表并解析出 Python 列表？输出解析器 outputparser 主要做两方面事情。第一，它能给模型下指令，要求模型按照指定的格式输出。第二，它能解析模型的输出，帮助我们从里面提取想要的信息。除了逗号分隔列表，json 也是一个很常见的输出格式，因为它非常方便解析。我们能很容易的把 JSON 字符串转换成字典列表或者是类实例。

假设我们经营着一个书籍点评网站，手头上有很多未经整理的书籍介绍，我们希望让 AI 根据介绍提取出书籍的书名、作者以及题材，那我们之后就可以把这些信息分别展现在网站前端了，因此我们会希望 AI 返回的结果符合预期格式，这要求在 AI 的输出里，无论是字段的名字还是值的类型，都应该和我们预期的相匹配。如果 AI 生成的 JSON 里面把 genres 叫做 genre，那我们用代码就提取不出来 genres，如果它对应的值的类型不是字符串列表，也会造成解析失败。

那如何利用 langchain 的解析器获得我们想要的 JSON 输出并解析出来呢？在 langchain 的 outputparsers 模块里，有一个叫 PydanticOutputParser 的解析器，它可以用于指挥 AI 输出符合格式要求的 JSON，并且帮我们进行解析。Pydantic 其实是一个比较常用的 Python 库的名字，负责数据解析和验证，它会利用 Python 类型提示来验证数据，确保数据满足特定的格式和类型要求，比如帮我们验证 genres 这个字段存在，并且对应的值确实是字符串组成的列表等等。

因此我们不只需要导入 langchain 的 PydanticOutputParser，如果你之前没有安装过 Pydantic 库，需要先进行安装，然后导入 Pydantic 库的一些东西，包括 base model 和 field。

```python

pip install pydantic

```

```python

from langchain_core.pydantic_v1 import BaseModel, Field

```

base model 是用于创建数据模式的，数据模式可以浅显的理解为数据的说明书，而 field 意思是字段，是用于为 base model 里的属性提供额外信息和验证条件的，具体用法后续会接触到。我们可以新建一个表示书籍信息的类，比如叫 book info，然后为了后续进行解析和验证，这个类需要继承自 base model。

接下来我们可以定义 book info 里面的字段了，字段名后面跟上冒号类型名可以用于指定字段的类型，book name 书名以及 author name 作者都是字符串，那它们对应的类型名叫 str，而 genres 书籍体裁是列表。由于列表属于复合类型，里面可以包含多个元素，所以要表示出它的类型，得从 typing 库里引入大写开头 list 作为类型名，然后方括号里面再放入列表元素的类型 str，然后我们还可以利用 field 函数为各个字段补充描述信息，那这个描述信息在后续也会传递给 AI，帮助它理解每一个字段的意思，所以并不是只为了给我们自己看。

```python

class BookInfo(BaseModel):
    book_name: str = Field(description="书籍的名字", example="百年孤独")
    author_name: str = Field(description="书籍的作者", example="加西亚·马尔克斯")
    genres: List[str] = Field(description="书籍的体裁", example=["小说", "文学"])

```

在我们借助 Pydantic 定义好希望解析出的数据长成什么样后，我们需要把它传给 Lang chain 的 PydanticOutputParser，创建一个 PydanticOutputParser 的实例，把 pandantic object 参数赋值为前面定义的继承自 base model 的 bookinfo 类。那么这个解析器能帮助我们做两件事情，第一从 AI 那里得到各个字段都符合格式和内容要求的 JSON，第二，把模型输出的 JSON 字符串解析成 bookinfo 实例，要查看解析器实际下达的指令，仍然是调用 get format instructions 方法。

```python

output_parser = PydanticOutputParser(pydantic_object=BookInfo)

```

可以看到解析器在教导 AI 要根据要求的数据模式输出对应的 JSON 结果，并且附上了我们自己定义的数据模式，所以我们只需要让 PydanticOutputParser 知道想要的 JSON 长成什么样就行了，那指导 AI 的脏活就交给解析器。下一步我们用 chat prompt template 消息提示模板把给 AI 的消息列表写出来，第一条消息是系统提示，后续跟上用户提示，那除了要填入的来自解析器的指令，我们也可以在系统消息里面补充其他指令，然后就可以把解析器的指令连带着用户提示里的变量值一起传给提示模板，调用模板的 invoke 方法得到最终的提示，把这个提示传给模型的 invoke 方法运行后，可以看到这个输出结果完全符合我们的要求，字段数量和名字与 bookinfo 类里定义的一致，各个值的数据类型也一致。

```python

prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions} 你输出的结果请使用中文。"),
    ("human", "请你帮我从书籍概述中，提取书名、作者，以及书籍的体裁。书籍概述会被三个#符号包围。\n###{book_introduction}###")
])book_introduction = """《明朝那些事儿》，作者是当年明月。2006年3月在天涯社区首次发表，2009年3月21日连载完毕，边写作边集结成书出版发行，一共7本。《明朝那些事儿》主要讲述的是从1344年到1644年这三百年间关于明朝的一些故事。以史料为基础，以年代和具体人物为主线，并加入了小说的笔法，语言幽默风趣。对明朝十六帝和其他王公权贵和小人物的命运进行全景展示，尤其对官场政治、战争、帝王心术着墨最多，并加入对当时政治经济制度、人伦道德的演义。它以一种网络语言向读者娓娓道出三百多年关于明朝的历史故事、人物。其中原本在历史中陌生、模糊的历史人物在书中一个个变得鲜活起来。《明朝那些事儿》为读者解读历史中的另一面，让历史变成一部活生生的生活故事。"""final_prompt = prompt.invoke({
    "book_introduction": book_introduction,
    "parser_instructions": output_parser.get_format_instructions()
})

```

这还没有完，由于 JSON 本质上是字符串，从字符串里提取信息很麻烦，但从类实例里提取对应信息很简单，那么我们可以继续借助 PydanticOutputParser 的解析能力，调用 parse 方法直接传入模型的回应，然后我们就得到了 book info。PydanticOutputParser 帮我们把模型的回复直接解析成了 book info 的一个实例，这样我们可以轻而易举地提取出里面的任何信息了。

```python

result = output_parser.invoke(response)

```

完整运行

```python

from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class BookInfo(BaseModel):
    book_name: str = Field(description="书籍的名字", example="百年孤独")
    author_name: str = Field(description="书籍的作者", example="加西亚·马尔克斯")
    genres: List[str] = Field(description="书籍的体裁", example=["小说", "文学"])

output_parser = PydanticOutputParser(pydantic_object=BookInfo)

print(output_parser.get_format_instructions())

prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions} 你输出的结果请使用中文。"),
    ("human", "请你帮我从书籍概述中，提取书名、作者，以及书籍的体裁。书籍概述会被三个#符号包围。\n###{book_introduction}###")
])

book_introduction = """《明朝那些事儿》，作者是当年明月。2006年3月在天涯社区首次发表，2009年3月21日连载完毕，边写作边集结成书出版发行，一共7本。《明朝那些事儿》主要讲述的是从1344年到1644年这三百年间关于明朝的一些故事。以史料为基础，以年代和具体人物为主线，并加入了小说的笔法，语言幽默风趣。对明朝十六帝和其他王公权贵和小人物的命运进行全景展示，尤其对官场政治、战争、帝王心术着墨最多，并加入对当时政治经济制度、人伦道德的演义。它以一种网络语言向读者娓娓道出三百多年关于明朝的历史故事、人物。其中原本在历史中陌生、模糊的历史人物在书中一个个变得鲜活起来。《明朝那些事儿》为读者解读历史中的另一面，让历史变成一部活生生的生活故事。"""

final_prompt = prompt.invoke({
    "book_introduction": book_introduction,
    "parser_instructions": output_parser.get_format_instructions()
})

model = ChatOpenAI(model="gpt-3.5-turbo")

response = model.invoke(final_prompt)

print(response.content)

result = output_parser.invoke(response)
result
result.book_name
result.genres

```

## 易懂版

angChain 的输出解析器（outputparser）核心就干两件特别实用的事：

一是给 AI 定规矩，让它严格按我们想要的格式输出内容；

二是帮我们解析 AI 的输出，直接提取出能在 Python 里用的信息。

日常用得最多的格式除了逗号分隔的列表，就是 JSON 了，因为 JSON 特别好解析，能轻松转成 Python 里的字典、列表，甚至是我们自定义的类对象。

举个实际的例子：如果我们运营一个书籍点评网站，手里有一堆乱糟糟的书籍介绍，想让 AI 从中提取书名、作者、书籍体裁，并把这些信息展示在网站上，就必须让 AI 的输出格式完全符合我们的要求 —— 不仅字段名要一模一样（比如 AI 要是把 “genres（体裁）” 写成 “genre”，代码就提不出数据了），字段的类型也不能错（比如体裁要求是列表，AI 只给了单个字符串，也会解析失败）。

接下来就一步步讲，怎么用 LangChain 里的PydanticOutputParser，让 AI 输出符合要求的 JSON，还能直接把 JSON 解析成 Python 里能用的内容，全程不省略任何关键步骤。

## 先搞懂核心工具：Pydantic

Pydantic 是 Python 的一个常用库，核心功能是数据解析和验证，简单说就是按我们定的规则检查数据：比如有没有指定的字段、字段值的类型对不对。而 LangChain 的 PydanticOutputParser，就是基于这个库做的解析器，既能指挥 AI 输出符合 Pydantic 规则的 JSON，又能直接把 AI 的 JSON 输出解析成 Python 类实例。

第一步先安装 Pydantic 库，在终端执行下面的命令就行：

```bash
pip install pydantic
```

## 第一步：导入需要的所有工具

从对应的库中导入后续要用的模块，每一个的作用后面会通俗讲，先按代码导入就行：

```python
# 导入列表类型，用来定义"体裁"这种列表字段
from typing import List

# 导入LangChain的Pydantic输出解析器
from langchain.output_parsers import PydanticOutputParser

# 导入LangChain的聊天提示模板，用来给AI写指令
from langchain.prompts import ChatPromptTemplate

# 导入Pydantic的核心模块，用来定义数据格式
from langchain_core.pydantic_v1 import BaseModel, Field

# 导入os，用来从环境变量读取API密钥（更安全）
import os

# 导入ChatOpenAI，用来调用大模型
from langchain_openai import ChatOpenAI
```

## 第二步：定义 AI 的输出格式（给 AI 定 "数据规矩"）

我们需要用 Pydantic 的BaseModel创建一个数据格式类，这个类就像给 AI 的 "数据说明书"，告诉它必须输出哪些字段、每个字段是什么类型、代表什么意思。

这里我们创建一个表示书籍信息的类BookInfo，并继承BaseModel，只有继承它，后续解析器才能验证和解析数据。

### 关键说明：

BaseModel：就是用来定 “数据说明书” 的核心，告诉解析器和 AI，我们要的是啥样的数据；

Field：给每个字段加描述和示例，这些内容会直接传给 AI，让 AI 明白每个字段要填什么，不只是给我们自己看的；

字段类型：书名、作者是单个字符串，用str；体裁是多个内容，用List[str]表示字符串组成的列表。

代码如下，复制就能用：

```python
class BookInfo(BaseModel):
    # 书籍名字，字符串类型，Field里写清楚描述和示例
    book_name: str = Field(description="书籍的名字", example="百年孤独")
    # 书籍作者，字符串类型
    author_name: str = Field(description="书籍的作者", example="加西亚·马尔克斯")
    # 书籍体裁，字符串组成的列表类型
    genres: List[str] = Field(description="书籍的体裁", example=["小说", "文学"])
```

## 第三步：创建解析器实例（让解析器知道规矩）

把我们刚定义的BookInfo数据格式传给PydanticOutputParser，创建一个解析器对象。这个解析器会自动完成两件事：

生成给 AI 的格式指令，让 AI 按BookInfo的规则输出 JSON；

后续直接解析 AI 的 JSON 输出，转成BookInfo类的实例。

代码超简单，就一行：

```python
# 创建解析器实例，传入我们定义的书籍数据格式
output_parser = PydanticOutputParser(pydantic_object=BookInfo)
```

如果想看看解析器到底给 AI 下了什么格式指令，执行print(output_parser.get_format_instructions())就能看到，相当于直接看解析器给 AI 的 “格式要求纸条”。

## 第四步：写 AI 的提示模板（给 AI 说清楚要做什么）

用 LangChain 的ChatPromptTemplate写给 AI 的对话指令，分成系统消息和人类消息两部分，就像我们和 AI 聊天时，先定规则，再提具体要求。

### 模板说明：

系统消息：传入解析器的格式指令，再加上 “输出结果用中文” 的要求，让 AI 严格按规矩来；

人类消息：明确告诉 AI 要做的事 —— 从书籍概述里提取信息，并且把书籍概述用###包围（方便 AI 识别内容边界）；

模板里的{}是变量，后续会传入实际的书籍概述和解析器指令。

代码如下：

```python
# 构建提示模板，分系统消息和人类消息
prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions} 你输出的结果请使用中文。"),
    ("human", "请你帮我从书籍概述中，提取书名、作者，以及书籍的体裁。书籍概述会被三个#符号包围。\n###{book_introduction}###")
])

# 准备要提取信息的书籍概述（以《明朝那些事儿》为例）
book_introduction = """《明朝那些事儿》，作者是当年明月。2006年3月在天涯社区首次发表，2009年3月21日连载完毕，边写作边集结成书出版发行，一共7本。《明朝那些事儿》主要讲述的是从1344年到1644年这三百年间关于明朝的一些故事。以史料为基础，以年代和具体人物为主线，并加入了小说的笔法，语言幽默风趣。对明朝十六帝和其他王公权贵和小人物的命运进行全景展示，尤其对官场政治、战争、帝王心术着墨最多，并加入对当时政治经济制度、人伦道德的演义。它以一种网络语言向读者娓娓道出三百多年关于明朝的历史故事、人物。其中原本在历史中陌生、模糊的历史人物在书中一个个变得鲜活起来。《明朝那些事儿》为读者解读历史中的另一面，让历史变成一部活生生的生活故事。"""

# 给模板传入实际变量，生成最终给AI的完整提示
final_prompt = prompt.invoke({
    "book_introduction": book_introduction,  # 实际的书籍概述
    "parser_instructions": output_parser.get_format_instructions()  # 解析器的格式指令
})
```

## 第五步：配置并调用自己的大模型（让 AI 按要求干活）

这里用你自己的豆包 OpenAI 兼容接口模型来配置，替换掉原有的通用模型配置，重点保留你的 base_url、api_key、model 参数，其他参数（随机性、最大 token 数等）也按你的设置来，这样就能调用自己的个人模型了。

### 关键说明：

base_url：豆包 OpenAI 兼容接口的固定地址，不用改；

api_key：用os.getenv("ARK_API_KEY")从环境变量读取，比直接写字符串更安全，也可以直接写成api_key="你的密钥字符串"；

其他参数：temperature（随机性）、max_tokens（AI 最大输出字数）等，按你的需求调整就行。

代码如下：

```python
# 配置你自己的豆包OpenAI兼容接口模型
chat_model = ChatOpenAI(
    # 豆包OpenAI兼容接口地址（固定不变）
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 你的豆包API密钥，从环境变量读取更安全，也可直接写api_key="你的密钥"
    api_key=os.getenv("ARK_API_KEY"),
    # 你自己的模型名称
    model="ep-m-20260212020325-2bpp5",
    temperature=1.2,  # 随机性0-2，值越高AI回复越多样
    max_tokens=300,  # AI回复的最大token数量（相当于最大字数）
    frequency_penalty=1.5,  # 频率惩罚，减少AI回复的重复内容
    top_p=0.8  # 核采样，辅助控制AI回复的多样性
)

# 调用模型，传入最终的提示，得到AI的回复
response = chat_model.invoke(final_prompt)

# 打印AI的原始输出（能看到标准的JSON格式内容）
print(response.content)
```

执行这部分代码后，AI 会输出一个符合BookInfo规则的 JSON 字符串，字段名、类型都和我们定的一模一样。

## 第六步：解析 AI 的输出（直接提取能用的信息）

AI 的输出本质是JSON 字符串，如果自己写代码解析字符串会很麻烦，而我们创建的解析器可以直接调用invoke方法，把 AI 的回复传进去，就能直接解析成BookInfo类的实例。

变成实例后，我们只需要用 「实例。字段名」 的方式，就能轻松提取出书名、作者、体裁，不用再处理字符串了，这也是解析器最核心的便利之处。

```python
# 用解析器解析AI的回复，得到BookInfo类的实例
result = output_parser.invoke(response)

# 直接提取信息，想拿哪个字段就用「实例.字段名」
result.book_name  # 提取书名
result.author_name  # 提取作者
result.genres  # 提取体裁（列表格式）
```

比如这次解析《明朝那些事儿》，result.book_name会直接返回明朝那些事儿，result.genres会返回["历史", "通俗文学"]这类符合要求的列表。

## 完整可运行代码（整合所有步骤，直接复制用）

把上面所有步骤整合，注释也标清楚，替换成你的 API 密钥就能直接运行，全程不用改其他内容：

```python
# 1. 导入所有需要的模块
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import os
from langchain_openai import ChatOpenAI

# 2. 定义书籍信息的数据格式（给AI定规矩）
class BookInfo(BaseModel):
    book_name: str = Field(description="书籍的名字", example="百年孤独")
    author_name: str = Field(description="书籍的作者", example="加西亚·马尔克斯")
    genres: List[str] = Field(description="书籍的体裁", example=["小说", "文学"])

# 3. 创建Pydantic输出解析器实例
output_parser = PydanticOutputParser(pydantic_object=BookInfo)

# 可选：打印解析器给AI的格式指令
# print(output_parser.get_format_instructions())

# 4. 构建提示模板并传入实际的书籍概述
prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions} 你输出的结果请使用中文。"),
    ("human", "请你帮我从书籍概述中，提取书名、作者，以及书籍的体裁。书籍概述会被三个#符号包围。\n###{book_introduction}###")
])

# 待提取的书籍概述
book_introduction = """《明朝那些事儿》，作者是当年明月。2006年3月在天涯社区首次发表，2009年3月21日连载完毕，边写作边集结成书出版发行，一共7本。《明朝那些事儿》主要讲述的是从1344年到1644年这三百年间关于明朝的一些故事。以史料为基础，以年代和具体人物为主线，并加入了小说的笔法，语言幽默风趣。对明朝十六帝和其他王公权贵和小人物的命运进行全景展示，尤其对官场政治、战争、帝王心术着墨最多，并加入对当时政治经济制度、人伦道德的演义。它以一种网络语言向读者娓娓道出三百多年关于明朝的历史故事、人物。其中原本在历史中陌生、模糊的历史人物在书中一个个变得鲜活起来。《明朝那些事儿》为读者解读历史中的另一面，让历史变成一部活生生的生活故事。"""

# 生成最终给AI的完整提示
final_prompt = prompt.invoke({
    "book_introduction": book_introduction,
    "parser_instructions": output_parser.get_format_instructions()
})

# 5. 配置并调用自己的豆包模型
chat_model = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.getenv("ARK_API_KEY"),  # 也可直接写api_key="你的豆包API密钥"
    model="ep-m-20260212020325-2bpp5",
    temperature=1.2,
    max_tokens=300,
    frequency_penalty=1.5,
    top_p=0.8
)

# 调用模型得到回复
response = chat_model.invoke(final_prompt)

# 打印AI的原始JSON输出
print("AI的原始输出：\n", response.content)

# 6. 解析AI输出并提取信息
result = output_parser.invoke(response)
print("\n提取的书名：", result.book_name)
print("提取的作者：", result.author_name)
print("提取的体裁：", result.genres)

```

## 拓展：逗号分隔列表的解析思路

其实解析逗号分隔列表和 JSON 的逻辑是一样的：

用 LangChain 对应的列表解析器（比如CommaSeparatedListOutputParser）；

解析器会给 AI 下指令，让它输出逗号分隔的内容；

调用解析器的方法，直接把 AI 的输出解析成 Python 的列表对象。

核心逻辑和 PydanticOutputParser 一致，都是 “解析器定格式 + 解析输出”，只是针对的格式不同而已。