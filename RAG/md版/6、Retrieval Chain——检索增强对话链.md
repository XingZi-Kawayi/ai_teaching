# 6、Retrieval Chain——检索增强对话链

## 一、整体流程总览

整个 RAG 问答系统的搭建，分为 6 个闭环步骤：

1. 知识库前置处理：文档加载→文本分块→向量化存储→生成检索器
2. 核心组件准备：初始化大模型实例、配置符合要求的对话记忆模块
3. 组装对话链：用官方封装方法，把三大核心组件拼接成完整的对话检索链
4. 调用对话问答：用固定格式触发对话，验证基础问答和连续对话记忆能力
5. 定制化功能拓展：开启源文档返回，排查 AI 幻觉、验证回答真实性
6. 方案优化说明：讲解当前默认逻辑的局限与优化方向

## 二、第一步：RAG 前置基础工作（文档加载→分块→向量化→检索器）

### 核心原理

要实现 RAG 问答，必须先完成前置工作：把你的知识库文档分割成语义完整的文本块，再将文本块转为向量存入向量数据库，最终得到一个能根据用户问题快速匹配相关内容的**检索器**。这是整个对话链能正常工作的前提，没有这一步，RAG 就无从谈起。

### 对应代码逐行精讲

#### 1. 导入所有依赖工具

先把整个流程需要的工具全部导入，每个工具的作用一目了然：

```python
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
```

#### 2. 加载知识库文档

把你的知识库内容放在本地demo2.txt文件中，通过加载器读取到程序里：

```python
loader = TextLoader("./demo2.txt")
docs = loader.load()
```

#### 3. 文本分块（核心细节）

长文档无法直接向量化和检索，必须切成小块，这里用的递归字符分割器，能最大程度保证语义完整：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
)
texts = text_splitter.split_documents(docs)
```

#### 4. 文本向量化 + 构建向量库 + 生成检索器

向量是文本的「数字身份证」，意思相近的文本，向量相似度也极高。用户提问时，系统会把问题也转为向量，快速在向量库中找到最相关的文本块，这个能力由检索器提供：

```python
embeddings_model = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings_model)
retriever = db.as_retriever()
```

至此，对话链三大前置准备的第一项「可用的文档检索器」就完成了。

## 三、第二步：完成剩余两大核心前置组件准备

### 核心原理

使用对话检索链前，必须完成三项核心前置准备：

1. 已完成文档切割与向量化，得到可用的检索器（上一步已完成）
2. 创建好聊天大模型实例（RAG 的核心生成组件）
3. 配置好连续对话所需的记忆模块（必须严格匹配 3 项关键参数）

### 对应代码逐行精讲

#### 1. 初始化聊天大模型实例

大模型是 RAG 架构的核心，负责理解用户问题、检索到的文档内容和历史对话，最终生成通顺准确的回答：

```python
model = ChatOpenAI(model="gpt-3.5-turbo")
```

#### 2. 配置对话记忆模块（新手最易踩坑的重点）

文档明确要求，记忆模块必须完成 3 项关键设置，否则对话链无法正常读取和更新对话历史，无法实现连续对话：

```python
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="answer"
)
```

通俗理解：这三个参数就像快递的收件信息，必须和对话链的「收件地址」完全一致，否则对话链既读不到之前的历史对话，也没法把新的对话存进去，连续对话就会失效。

## 四、第三步：正式创建对话检索链

### 核心原理

完成所有核心组件准备后，调用对话检索链的from_llm方法，给三大核心参数赋值，就能一键搭建好具备 RAG 能力、带对话记忆的对话链，无需手动拼接检索内容和用户问题。

### 对应代码精讲

```python
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory
)
```

这一步完成后，一个完整的 RAG 问答系统就搭建好了。它的自动工作流是：

1. 用户提问题
2. 检索器从知识库找相关文本
3. 把历史对话、用户问题、检索内容自动拼接
4. 传给大模型生成回答
5. 自动把本轮对话存入记忆模块

全程无需手动操作。

## 五、第四步：调用对话链，验证问答与记忆能力

### 核心原理

对话链通过invoke方法调用，必须传入字典格式的参数，且固定键名只能是chat_history和question，不能修改。

invoke执行后会返回字典结果，answer键对应 AI 生成的回答，question键对应用户的提问。

本轮对话会自动加入记忆模块，可通过连续提问验证记忆能力：模型能通过历史对话，理解用户提问中模糊主语的指代。

### 对应代码精讲

#### 1. 基础单轮问答调用

```python
question = "卢浮宫这个名字怎么来的？"
result = qa.invoke({"chat_history": memory, "question": question})
print(result["answer"])
```

执行后，对话链会先从知识库中检索「卢浮宫名字由来」的相关内容，再结合问题生成准确回答，同时自动把本轮的提问和回答存入记忆模块。

#### 2. 连续对话，验证记忆能力

```python
question = "它名字对应的拉丁语是什么？"
result2 = qa.invoke({"chat_history": memory, "question": question})
print(result2["answer"])
```

这里就能验证记忆能力：用户的问题里没有提到「卢浮宫」，但 AI 依然能理解用户问的是「卢浮宫名字对应的拉丁语」，就是因为对话链通过记忆模块，把上一轮的对话历史一起传给了大模型，大模型能精准理解上下文的指代关系，实现真正的连续对话，而非单轮问答。

## 六、第五步：定制化拓展 —— 开启源文档返回，排查 AI 幻觉

### 核心原理

最常用的定制化能力，是让对话链返回模型回答参考的源文档片段，只需在创建对话链时，添加return_source_documents=True参数即可开启。开启后，返回的字典会新增source_documents字段，对应检索到的、传给模型的文本片段，排序越靠前，和用户查询的相关性越高。该功能可直接验证回答的真实性，核对内容是否有知识库依据，排查 AI 幻觉问题。

### 对应代码精讲

```python
# 重新创建对话链，开启返回源文档功能
qa_with_source = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# 执行问答
result_with_source = qa_with_source.invoke({
    "chat_history": memory,
    "question": "卢浮宫这个名字怎么来的？"
})

# 查看回答
print(result_with_source["answer"])

# 查看源文档
print(result_with_source["source_documents"])
```

执行后，除了answer回答内容，还能通过result_with_source["source_documents"]查看 AI 回答参考的所有源文档片段，直接核对 AI 有没有「胡说八道」，这是 RAG 系统保障回答准确性的核心功能。

## 完整代码回顾

```python
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory

# 1. 加载知识库文档
loader = TextLoader("./demo2.txt")
docs = loader.load()

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
)
texts = text_splitter.split_documents(docs)

# 3. 文本向量化 + 构建向量库 + 生成检索器
embeddings_model = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings_model)
retriever = db.as_retriever()

# 4. 初始化大模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 5. 配置对话记忆
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="answer"
)

# 6. 创建对话检索链
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory
)

# 7. 执行问答
question = "卢浮宫这个名字怎么来的？"
result = qa.invoke({"chat_history": memory, "question": question})
print(result["answer"])

# 8. 连续对话验证记忆
question2 = "它名字对应的拉丁语是什么？"
result2 = qa.invoke({"chat_history": memory, "question": question2})
print(result2["answer"])
```
