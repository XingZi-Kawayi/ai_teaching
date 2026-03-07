# 1、引入langchain框架

模型是AI应用的核心，提供语言的理解和生成能力，langchain允许集成多种不同的模型大类别，主要分为两类。llm语言模型和chat model聊天模型。llm和chat model本质上都是大语言模型，不同之处在于llm是做文本补全的模型，而chat model是那些在对话方面进行了调优的模型。由于一个擅长补全，一个擅长对话，对于这两类模型，我们主要使用的接口是不太一样的，也就是他们接收和返回数据的方式有所区别。

具体来说，llm接收一个字符串儿作为输入，然后返回一个字符串儿作为输出，而chat model接受一个消息列表作为输入，然后返回一个消息作为输出。我们熟悉的chat-gpt、豆包、通义千问等都属于chat models，当前chat models可能是比llm更先进的选择，毕竟在对话方面效果更好，能提升用户的体验。langchain集成了多种不同的聊天模型，我们来一起看看如何通过langchain而不是原生API得到来自AI模型的回复。

首先，安装langchain与open AI集成的库，这个库名叫做langchain_opena在终端输入：

```bash
pip install langchain_openai
```

然后从langchain_openai里导入Open AI聊天模型。

```python

from langchain_openai import ChatOpenAI

```

创建一个Openai聊天模型的实例。

```python

model = ChatOpenAI(model="gpt-3.5-turbo")

```

可以直接在里面调整大模型的参数。

```python

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=1.2,
    max_tokens=300,
    model_kwargs={
        "frequency_penalty": 1.5
    }
)

```

由于温度和最大token数是比较常用的参数，可以直接作为可选参数进行设置，而频率惩罚等没那么常用的参数可以键值对的形式放入字典，传给model_kwargs参数，它允许我们设置原生API里的其他参数。

想了解更多可调整的参数，可以查阅langchain官方文档。

https://docs.langchain.com/oss/python/langchain/overview

现在我们就有了OpenAI聊天模型的实例，但在给他发消息前，我们还需要了解一下模型接收消息列表的具体结构。消息列表里的消息：

SystemMessage系统消息表示对AI系统的指令。

HumanMessage表示来自人类的聊天消息。

AIMessage表示来自AI的聊天消息。

假如我们想给模型传的消息，列表里包含系统消息和一条人类消息的话，可以从 LangChain 的 schema.messages 模块里引入 HumanMessage 和 SystemMessage，这两个类可以帮我们分别创建系统消息和人类消息。

```python

from langchain.schema.messages import SystemMessage, HumanMessage

```

创建系统消息的方法可以是把 SystemMessage 构造函数的 content 参数赋值为消息文本，创建人类消息的方法可以是给 HumanMessage 构造函数的 content 参数赋值为消息文本。

```python

system_message = SystemMessage(content="请你作为我的物理助教，用通俗易懂的语言解释物理概念。")
human = HumanMessage(content="什么是波粒二象性？")

```

那我们就可以在python列表里分别放入这两个创建出的消息组合出一个消息列表来。

```python

messages = [
    SystemMessage(content="请你作为我的物理助教，用通俗易懂的语言解释物理概念。"),
    HumanMessage(content="什么是波粒二象性？"),
]

```

要得到回复，我们可以调用模型的invoke方法，把消息列表作为参数传入。

```python

response = model.invoke(messages)

```

前面提到过聊天模型的输入是消息列表，输出是消息，大家可以看到这个输出的类型，不出意外是AI message表示来自AI的聊天消息。然后消息文本在AI message的content属性里，所以打印出来就可以看到来自AI的排版好的消息内容了。

langchain community这个库还集成了很多其他的聊天模型。如果感兴趣的话，也可以申请其他AI服务的API密钥，然后切换不同的模型来玩玩。