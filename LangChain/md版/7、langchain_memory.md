# 7、langchain_memory

AI 模型本身没有自带的上下文记忆能力，如果直接连续提问，比如先问 “丘吉尔是谁”，再问 “他是哪国人”，模型会因为不知道 “他” 的指代而无法回答。要实现带上下文的多轮对话，核心原理就是把每一轮的历史对话储存起来，下一轮提问时将历史对话和新问题一起传给模型。

LangChain 的memory模块为我们封装好了对话记忆的功能，不用自己手动写储存逻辑，下面我们结合完整代码，一步步讲清楚如何实现 AI 多轮对话，全程不修改核心代码，把每个步骤的作用和细节讲透。

## 第一步：导入所需核心模块

首先要导入实现对话记忆、提示模板、对接大模型的关键类，代码和作用对应如下：

```python

# 导入对话记忆核心类，用于储存历史对话
from langchain.memory import ConversationBufferMemory

# 导入聊天提示模板、消息列表占位符（核心，为历史对话留位置）
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 导入对接OpenAI聊天模型的类
from langchain_openai import ChatOpenAI

```

## 第二步：初始化对话记忆

使用ConversationBufferMemory创建记忆实例，这里有一个必设的关键参数return_messages=True，作用是让记忆里储存的历史对话以消息列表的形式存在，而非拼接成一整段字符串，方便后续和提示模板结合；如果不设这个参数，历史对话会变成一坨字符串，模型处理起来会很麻烦。

```python
# 初始化对话记忆，指定返回消息列表格式
memory = ConversationBufferMemory(return_messages=True)

# 查看当前记忆内容，传入空字典即可，刚初始化时记忆为空
memory.load_memory_variables({})
```

此时调用load_memory_variables({})，会发现返回结果里的history对应一个空列表，说明还没有储存任何历史对话。

## 第三步：手动保存历史对话（演示用）

为了测试记忆功能，我们可以先用save_context方法手动储存几轮历史对话，这个方法是 LangChain 封装的对话保存核心方法，需要传入两个字典作为参数：

第一个字典：键为input，值是用户的输入内容

第二个字典：键为output，值是AI 的回复内容

每调用一次save_context，就会保存一轮对话，我们连续保存两轮演示：

```python
# 保存第一轮对话：用户说名字，AI回应
memory.save_context({"input": "我的名字是风轻云淡夺魁"}, {"output": "你好，风轻云淡夺魁"})
# 查看记忆，此时history里已有第一轮对话
memory.load_memory_variables({})
# 保存第二轮对话：用户说职业，AI回应
memory.save_context({"input": "我是一名程序员"}, {"output": "好的，我记住了"})
# 再次查看记忆，history里有两轮对话了
memory.load_memory_variables({})
```

每次保存后调用load_memory_variables({})，都能看到history列表里新增了对应的用户和 AI 消息，这就是历史对话的储存过程。

## 第四步：构建带历史对话占位的提示模板

要让模型能看到历史对话，必须在提示模板里为历史对话留好位置，而且模板的消息顺序有严格要求：系统消息放第一条（定义模型的角色），然后是历史对话，最后是当前用户的新问题。

因为历史对话是消息列表，不是单个字符串，所以不能用普通的{变量名}占位，必须用MessagesPlaceholder这个专门的消息列表占位符，它需要传一个必选参数variable_name，用来指定消息列表的变量名，这里我们设为history，因为后续从记忆里读取的历史对话，键名就是history，必须一一对应。

```python

# 构建聊天提示模板，严格按「系统消息-历史对话-新问题」排序
prompt = ChatPromptTemplate.from_messages(
    [
        # 第一条：系统消息，定义模型角色
        ("system", "你是一个乐于助人的助手。"),
        # 第二条：历史对话的占位符，变量名设为history
        MessagesPlaceholder(variable_name="history"),
        # 第三条：当前用户的新问题，普通占位符{user_input}
        ("human", "{user_input}"),
    ]
)# 可再次查看记忆，确认历史对话还在memory.load_memory_variables({})

```

## 第五步：定义对接的大模型

这一步和普通调用 OpenAI 模型没有区别，只需指定要使用的模型名称即可，这里用最常用的gpt-3.5-turbo，也可以换成gpt-4等其他模型：

```python
# 定义OpenAI聊天模型，指定模型版本
model = ChatOpenAI(model="gpt-3.5-turbo")
```

## 第六步：构建链并首次调用，验证记忆效果

LangChain 中用|符号可以快速把提示模板和大模型拼接成一个链（chain），这个链的作用是：自动把传入的参数填充到提示模板，再把填充好的内容传给模型，最后返回模型的回复。

我们先提一个和历史对话相关的新问题，然后从记忆里提取历史对话，和新问题一起传给链，验证模型是否能结合上下文回答：

```python
# 把提示模板和模型拼接成链
chain = prompt | model

# 定义当前用户的新问题：询问名字（历史对话里有储存）
user_input = "你知道我的名字吗？"

# 从记忆里提取历史对话，通过键名history获取消息列表
history = memory.load_memory_variables({})["history"]

# 调用链，传入新问题和历史对话两个参数
result = chain.invoke({
    "user_input": user_input,
    "history': history
})

# 打印模型的回复结果
result
```

此时模型会结合历史对话中 “用户名字是风轻云淡夺魁” 的信息，正确回答出你的名字，这就实现了带上下文的第一次对话。

## 第七步：保存新一轮对话到记忆（关键步骤，不能忘）

刚才完成的 “用户问名字 + AI 回答名字” 这轮新对话，还没有被储存到记忆里，如果不手动保存，下一轮提问时模型就无法调用这轮对话的信息。

我们还是用save_context方法保存，注意 AI 的回复内容需要从result.content中获取（链的返回结果里，content字段是模型的纯文本回复）：

```python
# 把新一轮的用户输入和AI回复保存到记忆
memory.save_context({"input": user_input}, {"output": result.content})
# 查看记忆，此时history里新增了这轮对话
memory.load_memory_variables({})
```

## 第八步：再次调用链，验证多轮记忆的连贯性

我们再提一个更考验上下文的问题：让模型说出上一个问题是什么，再次调用链，看模型是否能结合所有历史对话正确回答：

```python
# 定义新问题：让模型重复上一个问题
user_input = "根据对话历史告诉我，我上一个问题问你的是什么？请重复一遍"

# 再次从记忆里提取最新的历史对话（包含上一轮的问名对话）
history = memory.load_memory_variables({})["history"]

# 调用链，传入新问题和最新历史对话
result = chain.invoke({
    "user_input": user_input,
    "history": history
})

# 打印模型的回复结果
result
```

此时模型会准确说出上一个问题是 “你知道我的名字吗？”，说明多轮对话的记忆功能完全实现，模型能识别所有的历史对话上下文。

## 核心总结

LangChain 实现多轮对话的核心是 **ConversationBufferMemory**，它负责全程储存历史对话，通过save_context存、load_memory_variables取；

提示模板中必须用 **MessagesPlaceholder** 为历史对话留位置，且变量名要和记忆中提取的键名一致（本文为history）；

消息顺序不可乱：系统消息 > 历史对话 > 当前新问题，系统消息必须在第一条；

每轮对话结束后，一定要把新的用户输入和 AI 回复保存到记忆，否则下一轮会丢失上下文；

本文是手动构建带记忆的链，目的是理解底层原理，实际开发中 LangChain 还封装了现成的带记忆对话链，无需手动调用save_context和load_memory_variables，会更便捷。

```python

from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

memory = ConversationBufferMemory(return_messages=True)

memory.load_memory_variables({})

memory.save_context({"input": "我的名字是风轻云淡夺魁"}, {"output": "你好，风轻云淡夺魁"})

memory.load_memory_variables({})

memory.save_context({"input": "我是一名程序员"}, {"output": "好的，我记住了"})

memory.load_memory_variables({})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)

memory.load_memory_variables({})

model = ChatOpenAI(model="gpt-3.5-turbo")

chain = prompt | model

user_input = "你知道我的名字吗？"

history = memory.load_memory_variables({})["history"]

result = chain.invoke({
    "user_input": user_input,
    "history": history
})

result

memory.save_context({"input": user_input}, {"output": result.content})

memory.load_memory_variables({})

user_input = "根据对话历史告诉我，我上一个问题问你的是什么？请重复一遍"

history = memory.load_memory_variables({})["history"]

result = chain.invoke({
    "user_input": user_input,
    "history": history
})

result
```