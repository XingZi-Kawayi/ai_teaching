# 4、Text Embedding——嵌入向量（嵌入）

我们把文本分割成一个个文本块后，接下来的核心步骤就是**将文本块转换成向量**，这个过程就叫做**嵌入**。嵌入可不是随便转换，核心要求是：向量里要包含文本的语法、语义关系，比如意思相近的文本，转成的向量在向量空间里距离会更近，无关文本的向量距离则会更远。

这里要注意一个知识点：Bert 本身不能直接做嵌入，得借助专门的**嵌入模型**，像 OpenAI、百度等 AI 服务商都有现成的嵌入模型可用，接下来我们就以**OpenAI 的嵌入模型**为例，结合代码一步步讲清楚实操方法，所有核心代码保持原样，保证能直接跟着做。

## 一、前期准备工作

想用 OpenAI 的嵌入模型，必须先做好两件事，这是后续操作的基础：

1. 安装openai库（代码里会给出安装命令）；
2. 拥有 OpenAI 的 API 密钥：如果之前用过 GPT 模型，这两步大概率已经完成，没完成的话可以参考相关 GPT 模型的课程配置。

API 密钥的使用方式：要么把密钥储存到电脑的**环境变量**中（后续代码不用手动传密钥，更便捷），要么在代码里手动赋值（适合临时使用或课程专用 API）。

## 二、核心实操步骤（代码 + 详解）

所有操作都是基于 Python 实现，每一行代码的作用都会对应讲解，确保能看懂、能运行。

### 步骤 1：安装 openai 依赖库

如果你的环境里还没装openai，先执行下面的安装命令：

```bash
pip install openai
```

### 步骤 2：导入嵌入模型模块

我们不是直接用 OpenAI 原生的嵌入功能，而是用**LangChain 封装的OpenAIEmbeddings**（LangChain 是大模型开发工具库，封装后能更方便地和后续的向量数据库配合使用），导入代码如下：

```python
from langchain_openai import OpenAIEmbeddings
```

### 步骤 3：创建嵌入模型的实例

这一步是指定用哪个嵌入模型，同时处理 API 密钥的传入，核心是OpenAIEmbeddings类的实例化，有**两种使用场景**，对应不同的代码写法：

#### 场景 1：使用自己的 OpenAI API，且密钥已存到环境变量

直接指定model参数即可，不用手动传密钥，默认用text-embedding-3-large这个模型（OpenAI 的主流嵌入模型）：

```python
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
```

#### 场景 2：使用课程提供的专用 API

需要额外手动传入openai_api_key（你的课程 API 密钥）和openai_api_base（课程的 API 地址）：

```python
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key="<你的API密钥>",
    openai_api_base="https://api.aigc369.com/v1"
)
```

**关键**：创建好的embeddings_model实例是后续生成向量的核心，下一节讲向量储存时还会用到它。

### 步骤 4：调用方法生成嵌入向量

创建好模型实例后，通过 **embed_documents方法**就能把文本转成向量，这个方法需要传入**字符串列表**（一个字符串对应一个文本 / 文本块），返回的结果也是一个列表，我们结合代码看具体效果：

```python
embeded_result = embeddings_model.embed_documents(["Hello world!", "Hey bro"])
```

#### 对返回结果的解读（配套代码查看）

**查看返回结果的长度**：

```python
len(embeded_result)
```

结果会是**2**，因为我们传入了 2 个字符串（["Hello world!", "Hey bro"]），一个文本对应一个向量，所以返回列表的长度和传入的文本数量一致。

**直接查看嵌入结果**：

```python
print(embeded_result)
```

会看到一个嵌套列表，外层列表的每个元素还是一个**纯数字列表**，这个数字列表就是对应文本的**嵌入向量**，所有数字共同表征了文本的语义、语法信息。

**查看向量的维度**：

```python
len(embeded_result[0])
```

结果会是**3072**，因为text-embedding-3-large模型**默认返回 3072 维的嵌入向量**，维度就代表数字列表里有多少个数字。

### 步骤 5：自定义嵌入向量的维度

如果觉得默认的 3072 维维度太高，会增加后续存储和计算的成本，我们可以在创建模型实例时，通过 **dimensions参数** 手动指定更小的维度，比如 1024 维：

```python
# 重新创建指定维度的模型实例
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

# 生成嵌入向量
embeded_result = embeddings_model.embed_documents(["Hello world!", "Hey bro"])

# 查看新的向量维度
len(embeded_result[0])
```

运行后len(embeded_result[0])的结果会是**1024**，说明已经成功将向量维度改成了 1024 维，维度可以根据自己的需求灵活调整。

## 完整代码回顾

```python
# 安装依赖
# pip install openai

from langchain_openai import OpenAIEmbeddings

# 创建嵌入模型实例
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# 生成嵌入向量
embeded_result = embeddings_model.embed_documents(["Hello world!", "Hey bro"])

# 查看结果
print(f"结果数量: {len(embeded_result)}")
print(f"向量维度: {len(embeded_result[0])}")
print(f"第一个向量: {embeded_result[0][:10]}...")  # 只显示前10个数字
```
