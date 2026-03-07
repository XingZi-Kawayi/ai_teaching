# 1+、LLM的调用——字节版

模型是 AI 应用的核心，提供语言的理解和生成能力，LangChain 允许集成多种不同的大模型类别，主要分为两类：LLM（语言模型）和 Chat Model（聊天模型）。LLM 和 Chat Model 本质上都是大语言模型，不同之处在于 LLM 是做文本补全的模型，而 Chat Model 是那些在对话方面进行了调优的模型。由于一个擅长补全，一个擅长对话，这两类模型使用的接口有所区别，也就是它们接收和返回数据的方式不同。

具体来说，LLM 接收一个字符串作为输入，返回一个字符串作为输出；而 Chat Model 接受一个消息列表作为输入，返回一个消息作为输出。豆包、ChatGPT、通义千问等都属于 Chat Models，当前 Chat Models 是比 LLM 更先进的选择，在对话场景下效果更好，能提升用户体验。LangChain 集成了多种聊天模型，下面我们看看如何通过 LangChain（而非原生 API）调用豆包大模型获取回复。

第一步：安装依赖库

国内大模型使用langchain有三种方式

基于 LangChain 基础接口自定义封装（最稳定）

使用国内社区维护的适配库，国内开发者针对 LangChain 做了专门的适配扩展，比官方更贴合国内模型

使用 LangChain 的 “通用 API 封装”，LangChain 提供了ChatOpenAI的兼容模式，国内很多模型支持 OpenAI 格式的 API（比如豆包、通义千问都有 OpenAI 兼容接口），可以直接复用：

现在使用方法三

```bash
#清华源
pip install langchain_openai python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple

#阿里源
pip install langchain_openai python-dotenv -i https://mirrors.aliyun.com/pypi/simple/
```

第二步：导入并创建豆包聊天模型实例

```python

import os# 导入LangChain的OpenAI兼容Chat模型（核心）from langchain_openai import ChatOpenAI# 导入消息模型（和你原来的代码一致）from langchain.schema.messages import SystemMessage, HumanMessage# 初始化豆包模型（通过OpenAI兼容接口）chat_model = ChatOpenAI(    # 1. 核心配置：豆包的OpenAI兼容接口地址（固定）    base_url="https://ark.cn-beijing.volces.com/api/v3",    # 2. API密钥：从环境变量读取（替换为你的豆包API Key）    api_key=os.getenv("ARK_API_KEY"),    #模型推理节点，替换为你选择的模型    model="ep-m-20260212020325-2bpp5",    temperature=1.2,    # 随机性，0-2之间，值越高回复越多样    max_tokens=300,     # 回复最大token数    frequency_penalty=1.5,  # 频率惩罚，减少重复内容    top_p=0.8           # 核采样，控制回复多样性)

```

关键参数说明：

api_key：豆包大模型的 API 密钥（必须配置，从字节跳动开放平台申请）；

temperature：常用参数，控制回复的随机性，0 为确定性回复，2 为最大随机性；

max_tokens：常用参数，限制模型返回的最大 token 数量，避免回复过长；

model_kwargs：用于传递豆包原生 API 的其他参数（如frequency_penalty频率惩罚、top_p核采样等），具体可参考豆包官方 API 文档。

第三步：构造消息列表

Chat Model 接收消息列表作为输入，消息类型包括：

SystemMessage：对 AI 的系统指令（定义 AI 的角色、行为准则）；

HumanMessage：人类用户的聊天消息；

AIMessage：AI 返回的消息（多用于多轮对话）。

构造包含系统指令和用户问题的消息列表：

```python

# 构造消息列表messages = [    SystemMessage(content="请你作为我的物理助教，用通俗易懂的语言解释物理概念。"),    HumanMessage(content="什么是波粒二象性？"),]

```

第四步：调用模型获取回复

通过模型的invoke方法传入消息列表，获取豆包的回复：

```python

# 调用模型获取回复response = model.invoke(messages)# 输出回复内容（AI回复的文本在content属性中）print(response.content)print(response.content)

```

第五步：运行

```python

import os# 导入LangChain的OpenAI兼容Chat模型（核心）from langchain_openai import ChatOpenAI# 导入消息模型（和你原来的代码一致）from langchain.schema.messages import SystemMessage, HumanMessage# 初始化豆包模型（通过OpenAI兼容接口）chat_model = ChatOpenAI(    # 1. 核心配置：豆包的OpenAI兼容接口地址（固定）    base_url="https://ark.cn-beijing.volces.com/api/v3",    # 2. API密钥：从环境变量读取（替换为你的豆包API Key）    api_key=os.getenv("ARK_API_KEY"),    model="ep-m-20260212020325-2bpp5",    temperature=1.2,    # 随机性，0-2之间，值越高回复越多样    max_tokens=300,     # 回复最大token数    frequency_penalty=1.5,  # 频率惩罚，减少重复内容    top_p=0.8           # 核采样，控制回复多样性)# 构造消息列表（物理助教场景，和你原来的需求一致）messages = [    SystemMessage(content="请你作为我的物理助教，用通俗易懂的语言解释物理概念，避免使用过于专业的术语，让初中生也能听懂。"),    HumanMessage(content="什么是波粒二象性？"),]# 调用模型获取回复（和你原来的调用方式完全一致）response = chat_model.invoke(messages)# 输出结果print("=== 波粒二象性的通俗解释 ===")print(response.content)