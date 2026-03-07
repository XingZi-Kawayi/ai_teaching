# 9、langchain_memory（3）

AI 模型本身没有自带的上文记忆能力，就像你问它 “丘吉尔是谁”，再问 “他是哪国人”，它会因为不知道 “他” 指谁而答不上来。想要实现多轮带上下文的对话，核心办法就是把历史对话存起来，每轮提问都把历史对话 + 新问题一起传给模型，模型回复后再把本轮对话也存进历史里，形成闭环。

LangChain 的memory模块已经帮我们封装好了各种记忆功能，不用自己手动写复杂的存储逻辑，既可以手动构建带记忆的对话链（搞懂底层原理），也可以直接用现成的对话链（实际开发超省事）。下面从最基础的记忆类型开始，一步步讲清楚原理和实操，代码和核心逻辑完全保留，保证能看懂、能运行。

### 先做基础准备：导入核心库 + 初始化模型

不管用哪种记忆类型，第一步都要导入需要的模块，再初始化大模型（这里用 OpenAI 的 gpt-3.5-turbo 为例），代码直接复制就能用：

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory
)

# 初始化大模型，后续所有记忆类型都基于这个模型
model = ChatOpenAI(model="gpt-3.5-turbo")
```

# 一、最基础的记忆：ConversationBufferMemory（完整记忆）

这是我们最开始用的记忆类型，特点是把所有对话一字不漏存起来，不丢任何信息，就像拿个小本本把和 AI 的聊天全部记下来，是理解其他记忆类型的基础。

### 核心细节：手动构建带记忆的链（搞懂底层原理）

手动做一遍能清楚知道 “记忆是怎么存、怎么用的”，核心分 5 步，每一步的作用都讲得明明白白：

初始化记忆：必须加return_messages=True，不然记忆里的内容会是一整坨字符串，而不是规整的消息列表，后续没法和模型配合；

查看记忆内容：用load_memory_variables({})方法，传入空字典，就能看到记忆里存的内容，刚初始化时是个空列表；

手动存对话：用save_context(用户输入, AI输出)方法，两个参数都是字典，把一轮对话存进记忆，存多少轮都可以反复调这个方法；

做提示模板：模板里要给 “历史对话” 留位置（用MessagesPlaceholder占位，变量名设为history，和记忆里的键名对应），且系统消息放第一条，历史对话放新问题前面，不然模型识别不了上下文；

构建链并调用：把提示模板、模型结合成链，传入新问题 + 记忆里的历史对话，模型回复后，再用save_context把本轮对话存进记忆，完成一次闭环。

简单说，手动构建的核心就是自己管 “存历史” 和 “取历史”，虽然麻烦，但能彻底搞懂记忆的工作逻辑。

### 实操更省事：用现成的 ConversationChain（实际开发首选）

LangChain 封装了ConversationChain，能自动帮我们做 “取历史、传模型、存新对话” 的工作，不用手动调load_memory_variables和save_context，代码超简单，效果就是 AI 能记住你的话：

```python
## ConversationBufferMemory 实操代码
memory = ConversationBufferMemory(return_messages=True)  # 初始化完整记忆
chain = ConversationChain(llm=model, memory=memory)  # 构建带记忆的现成对话链

# 第一轮对话：告诉AI你的名字
chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})

# 第二轮对话：问AI你的名字，它能通过记忆正确回答
chain.invoke({"input": "我的名字是什么？"})
```

运行效果：AI 会准确回答 “你的名字是粒粒”，因为所有对话都被完整存在记忆里了。

# 二、轻量化记忆：ConversationBufferWindowMemory（窗口记忆）

完整记忆会一直存所有对话，聊得越多存的内容越多，后续传给模型的信息就越杂。窗口记忆就是只记最近的 k 轮对话，像开了个 “小窗口”，只看窗口里的内容，超出的就丢掉，适合只需要短期上下文的场景。

### 核心参数：k

k表示要保留的最近对话轮数，比如 k=1，就只记最后 1 轮对话，前面的全部舍弃。

### 实操代码 + 效果

```python
## ConversationBufferWindowMemory 实操代码
memory = ConversationBufferWindowMemory(k=1, return_messages=True)  # 只保留最近1轮
chain = ConversationChain(llm=model, memory=memory)

chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})  # 第1轮：说名字
chain.invoke({"input": "我是一个程序员"})  # 第2轮：说职业
chain.invoke({"input": "我的名字是什么？"})  # 第3轮：问名字
```

运行效果：AI 会说 “没有相关信息”，因为 k=1，记忆里只保留了最后 1 轮 “我是一个程序员”，前面的名字被丢掉了。

# 三、省空间记忆：ConversationSummaryMemory（摘要记忆）

如果聊的内容特别多，就算存最近几轮，内容也会很长，传给模型时会消耗大量 token（相当于花钱）。摘要记忆不存原始对话，而是让 AI 把历史对话总结成一句 / 一段核心话，既保留关键信息，又大幅压缩内容，超省空间。

### 核心细节

初始化时必须加llm=model，因为需要让大模型自己做对话摘要，这是和前两种记忆的最大区别。

### 实操代码 + 效果

```python
## ConversationSummaryMemory 实操代码
memory = ConversationSummaryMemory(return_messages=True, llm=model)  # 传入模型做摘要
chain = ConversationChain(llm=model, memory=memory)

chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})  # 第1轮：说名字
chain.invoke({"input": "我是一个程序员，你呢？"})  # 第2轮：说职业+反问
chain.invoke({"input": "我的名字是什么？"})  # 第3轮：问名字
```

运行效果：AI 能准确回答 “你的名字是粒粒”，因为记忆里存了对话摘要，里面包含了 “名字、职业” 这些核心信息。

# 四、最实用的混合记忆：ConversationSummaryBufferMemory（摘要 + 窗口记忆）

这款是把窗口记忆和摘要记忆结合起来，兼顾 “精准度” 和 “省空间”，是实际开发中最常用的记忆类型，没有之一。

### 核心原理 + 参数

用max_token_limit设置token 阈值：

当历史对话的总 token 数没超过阈值：完整保留原始对话（保证短期对话的精准度）；

当历史对话的总 token 数超过阈值：把早期的对话总结成摘要，只保留最近的原始对话 + 早期摘要（既省空间，又不丢关键信息）。

初始化也需要加llm=model，因为需要模型生成早期对话的摘要。

### 实操代码 + 效果

```python
## ConversationSummaryBufferMemory 实操代码
memory = ConversationSummaryBufferMemory(llm=model, max_token_limit=100, return_messages=True)  # 阈值100token
chain = ConversationChain(llm=model, memory=memory)

chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})  # 第1轮：说名字
chain.invoke({"input": "我是一个程序员，你呢？"})  # 第2轮：说职业+反问
chain.invoke({"input": "我的名字是什么？我前面说过的"})  # 第3轮：带上下文问名字
```

运行效果：AI 能准确回答名字，因为当前对话的 token 数没超过 100，记忆里完整保留了原始内容；如果继续聊天，token 数超 100，AI 会自动把早期对话做成摘要。

# 五、按 token 截断的记忆：ConversationTokenBufferMemory（Token 窗口记忆）

和上面的混合记忆类似，也是按max_token_limit设置 token 阈值，但它只做一件事：保留总 token 数不超阈值的最近对话，超出的直接丢掉，不会生成任何摘要，比混合记忆更简单，适合对对话精准度要求不高、只想单纯省 token 的场景。

### 核心细节

初始化需要加llm=model，因为需要模型计算每段对话的 token 数，才能判断是否超出阈值。

### 实操代码 + 效果··

```python
## ConversationTokenBufferMemory 实操代码
memory = ConversationTokenBufferMemory(llm=model, max_token_limit=200, return_messages=True)  # 阈值200token
chain = ConversationChain(llm=model, memory=memory)

chain.invoke({"input": "你好，我的名字是粒粒"})  # 第1轮：说名字
chain.invoke({"input": "我是一个程序员，你呢？"})  # 第2轮：说职业+反问
chain.invoke({"input": "我的名字是什么？我前面说过的"})  # 第3轮：带上下文问名字
```

运行效果：AI 能准确回答名字，因为当前对话的 token 数远低于 200；如果后续对话超阈值，会从最早的对话开始丢，直到总 token 数符合要求。

## 最后总结：核心要点 + 开发建议

模型无原生记忆：所有多轮对话记忆，本质都是 “人工存历史对话，再传给模型”，LangChain 只是帮我们封装了这个过程；

手动构建 vs 现成链：手动构建是为了搞懂取历史→传模型→存新对话的底层逻辑，实际开发直接用ConversationChain，省时又不易错；

记忆类型可无缝替换：用现成链时，想换不同的记忆策略，只需要改初始化 memory 的代码，其他部分完全不用动，比如把完整记忆换成窗口记忆，只改一行memory = ...；

记忆类型选择技巧：

想完整存对话、不怕 token 消耗：用ConversationBufferMemory；

只需要短期上下文：用ConversationBufferWindowMemory；

对话多、想省 token：用ConversationSummaryMemory；

兼顾精准度和省空间（首选）：用ConversationSummaryBufferMemory；

简单按 token 截断、不做摘要：用ConversationTokenBufferMemory。

```python

from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory
)

model = ChatOpenAI(model="gpt-3.5-turbo")

# ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)
chain = ConversationChain(llm=model, memory=memory)
chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})
chain.invoke({"input": "我的名字是什么？"})

# ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1, return_messages=True)
chain = ConversationChain(llm=model, memory=memory)
chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})
chain.invoke({"input": "我是一个程序员"})
chain.invoke({"input": "我的名字是什么？"})

# ConversationSummaryMemory
memory = ConversationSummaryMemory(return_messages=True, llm=model)
chain = ConversationChain(llm=model, memory=memory)
chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})
chain.invoke({"input": "我是一个程序员，你呢？"})
chain.invoke({"input": "我的名字是什么？"})

# ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm=model, max_token_limit=100, return_messages=True)
chain = ConversationChain(llm=model, memory=memory)
chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})
chain.invoke({"input": "我是一个程序员，你呢？"})
chain.invoke({"input": "我的名字是什么？我前面说过的"})

# ConversationTokenBufferMemory
memory = ConversationTokenBufferMemory(llm=model, max_token_limit=200, return_messages=True)
chain = ConversationChain(llm=model, memory=memory)
chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})
chain.invoke({"input": "我是一个程序员，你呢？"})
chain.invoke({"input": "我的名字是什么？我前面说过的"})

```