# 8、langchain_memory(2)

我们日常和 AI 对话时，最常用的需求就是让 AI 记住之前的聊天内容（比如告诉它你的名字，后续它能准确叫出来）。如果自己手动写代码实现，要反复加载历史对话、保存新对话，非常繁琐。而 LangChain 专门给我们准备了开箱即用的ConversationChain，它天生适配对话场景，自带记忆管理，几行代码就能搞定带记忆的连续对话。

下面我们把代码和原理结合，一步步拆解，全程不修改核心代码，只用大白话讲透用法。

# 一、核心原理先搞懂

ConversationChain是 LangChain 封装好的专属对话链，核心解决两个痛点：

自动管理对话记忆：不用你手动写代码加载历史对话、保存新对话，调用后自动完成

一键打通模型 + 记忆 + 提示词：只需要把模型、记忆体（可选自定义提示词）传给它，就能直接用，不用自己拼接复杂逻辑

它的核心三要素：

LLM 模型：AI 的 “大脑”，负责生成回答

Memory 记忆体：AI 的 “笔记本”，负责存聊天历史

ConversationChain：把大脑和笔记本绑定的 “组装器”，自动完成记忆的读写和对话的流转

# 二、基础版：3 步实现带记忆的连续对话

先看最基础的用法，实现 “告诉 AI 名字，后续它能记住” 的效果，完整代码如下，我们逐行拆解：
```python
# 1. 导入需要的工具包
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 2. 创建AI"大脑"：指定用哪个大模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 3. 创建AI"笔记本"：记忆体，专门存聊天记录
memory = ConversationBufferMemory(return_messages=True)

# 4. 组装对话链：把大脑和笔记本绑定，做成完整的对话工具
chain = ConversationChain(llm=model, memory=memory)

# 5. 开始对话，直接调用invoke就行，全程不用手动管记忆
chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})
chain.invoke({"input": "我告诉过你我的名字，是什么？"})
```


### 逐行拆解 + 对应原理

导入工具包

ConversationChain：今天的主角，现成的对话链，帮我们管对话和记忆

ChatOpenAI：对接 OpenAI 大模型，就是 AI 的核心大脑

ConversationBufferMemory：最基础的记忆体，完整保存所有聊天历史

提示词相关的工具，基础版用不到，进阶自定义人设时会用到

创建 AI 大脑model = ChatOpenAI(model="gpt-3.5-turbo")这一步就是给对话指定 AI 模型，相当于给对话找了个聪明的大脑，这里用的是通用的 gpt-3.5-turbo 模型，可根据需求更换。

创建记忆体memory = ConversationBufferMemory(return_messages=True)这就是给 AI 开了个专属 “聊天记录本”，return_messages=True的意思是，存的内容是 AI 能直接读懂的对话格式，不用我们手动转换格式。划重点：这个记忆体和ConversationChain配合，会自动读写聊天记录，不用我们手动写代码加载、保存。

组装对话链chain = ConversationChain(llm=model, memory=memory)这是最核心的一步，把 AI 大脑和聊天记录本绑定，做成一个完整的带记忆对话工具。不用写任何额外逻辑，它天生就会：调用前自动加载历史对话，调用后自动把本轮对话存到记忆体里。

发起对话
```python
chain.invoke({"input": "你好，我的名字是风轻云淡夺魁"})
chain.invoke({"input": "我告诉过你我的名字，是什么？"})
```
对话只需要调用invoke方法，参数固定是{"input": "你要说的话"}。

第一句调用后，你的自我介绍和 AI 的回复，会自动存到记忆体里

第二句调用时，对话链会自动把历史对话喂给 AI，AI 就能准确说出你的名字，实现带记忆的对话

### 基础版核心优势

完全不用手动调用load_memory_variables（加载记忆）、save_context（保存对话），所有记忆操作全自动化，几行代码就能实现丝滑的连续对话，不用重复造轮子。

# 三、进阶版：自定义 AI 人设，给对话加提示模板

ConversationChain不止能实现基础的记忆对话，还支持自定义提示模板，给 AI 定专属人设（比如暴躁阴阳怪气、温柔贴心、专业客服等），完整代码如下：
```python
# 1. 自定义提示模板，给AI定人设
prompt = ChatPromptTemplate.from_messages([
    # system：给AI定的人设和规矩
    ("system", "你是一个脾气暴躁的助手，喜欢冷嘲热讽和用阴阳怪气的语气回答问题。"),
    # 固定写法：给历史对话留位置，变量名必须是history，不能改！
    MessagesPlaceholder(variable_name="history"),
    # 固定写法：用户输入的位置，变量名必须是input，不能改！
    ("human", "{input}")
])

# 2. 同样创建AI大脑和记忆体
model = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory(return_messages=True)

# 3. 组装对话链，多传一个prompt参数，把自定义人设传进去
chain = ConversationChain(llm=model, memory=memory, prompt=prompt)

# 4. 测试对话，AI会按人设回答，同时保留记忆功能
chain.invoke({"input": "今天天气怎么样？"})
chain.invoke({"input": "你记得我问的上一个问题不，是什么？"})
```
### 关键细节拆解（避坑重点）

提示模板的固定规则（必须遵守，否则会报错）文档里特别强调：ConversationChain有强制的变量名要求

用户输入的变量名必须是 input，对应模板里的{input}

历史对话的变量名必须是 history，对应MessagesPlaceholder(variable_name="history")这两个名字不能随便改，否则对话链找不到对应的内容，直接报错。

人设自定义方法模板里的system内容，就是给 AI 定的核心规矩，你可以随便修改：

比如改成 “你是一个温柔贴心的育儿助手，回答要通俗易懂，有耐心”

也可以改成 “你是一个专业的 Java 开发工程师，回答只讲技术干货，拒绝废话”只要把人设写在 system 里，AI 的所有回答都会遵循这个设定。

记忆功能依然生效哪怕加了自定义提示模板，对话链的自动记忆功能完全不受影响。比如上面的代码里，你先问天气，再问上一个问题是什么，AI 依然能准确记住，同时还会用暴躁阴阳的语气回答。

# 四、最后总结

ConversationChain是 LangChain 专为对话场景封装的现成工具，核心价值是自动化管理对话记忆，省去手动加载、保存历史对话的繁琐代码。

基础用法只需要 3 步：创建模型→创建记忆体→组装对话链，调用invoke就能直接实现带记忆的连续对话。

支持自定义提示模板给 AI 定人设，只需要遵守input和history两个固定变量名的规则，不会报错的同时，还能完全保留记忆功能。

代码里用到的ConversationBufferMemory只是最基础的记忆类型，它会完整保存所有聊天记录，LangChain 还有更多记忆类型可按需替换，不影响ConversationChain的核心用法。