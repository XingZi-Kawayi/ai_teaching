# 7、Documents Chain——把外部文档塞给模型的不同方式

## 一、核心概念与 4 种文档处理方式

在 RAG 问答系统中，当检索器从知识库中找到多个相关文本块后，需要把这些文本块和用户问题一起传给大模型，让模型基于这些外部文档生成回答。而如何把这些文档塞进模型的上下文窗口，**LangChain 提供了 4 种不同的策略**，每种策略各有优劣，对应不同的场景需求。

### 4 种文档处理方式对比

| 方式 | 核心逻辑 | 优点 | 缺点 | 适用场景 |
|------|----------|------|------|----------|
| stuff | 把所有文档直接塞进提示词，一次性传给模型 | 最简单、只调用一次模型 | 文档太长会超出上下文窗口限制 | 文档数量少、总长度短的场景 |
| map_reduce | 每个文档单独调用模型生成中间答案，再汇总成最终答案 | 可处理任意数量文档，不受上下文限制 | 需要多次调用模型，成本高、速度慢 | 文档数量多、需要全面参考的场景 |
| refine | 第一个文档生成初始答案，后续文档逐步迭代优化答案 | 类似 map_reduce，但能逐步优化，答案质量可能更高 | 同样需要多次调用，依赖迭代顺序 | 需要逐步精炼答案的复杂场景 |
| map_rerank | 每个文档单独调用模型打分并生成答案，选最高分答案返回 | 自动选出最相关文档的答案 | 不汇总多个文档的信息，可能遗漏内容 | 只需一个最准确答案的场景 |

## 二、前置准备：搭建完整的 RAG 基础环境

无论用哪种文档处理方式，都需要先完成 RAG 的基础流程：文档加载→文本分割→向量化存储→构建检索器→初始化大模型。这部分代码是 4 种方式共用的前置步骤。

### 1. 导入所有依赖工具

```python
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
```

### 2. 加载知识库文档

```python
loader = TextLoader("./demo2.txt")
docs = loader.load()
```

### 3. 文本分块

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
)
texts = text_splitter.split_documents(docs)
```

### 4. 构建向量库和检索器

```python
embeddings_model = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings_model)
retriever = db.as_retriever()
```

### 5. 初始化大模型

```python
model = ChatOpenAI(model="gpt-3.5-turbo")
```

### 6. 配置对话记忆

```python
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="answer"
)
```

## 三、4 种文档处理方式的代码实现

### 方式 1：stuff（默认方式）

stuff 是 ConversationalRetrievalChain 的默认方式，如果不指定 chain_type 参数，就是 stuff 方式。它的核心逻辑是：把所有检索到的文档直接拼接成一个字符串，塞进提示词里一次性传给模型。

#### 代码实现

```python
# stuff 方式（默认，无需指定 chain_type）
qa_stuff = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory
)

# 或者显式指定 chain_type="stuff"
qa_stuff = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    chain_type="stuff"
)

# 执行问答
result = qa_stuff.invoke({"chat_history": memory, "question": "卢浮宫这个名字怎么来的？"})
print(result["answer"])
```

#### 优缺点分析

- **优点**：实现最简单，只调用一次大模型，速度快、成本低
- **缺点**：如果检索到的文档总长度超过模型的上下文窗口限制，会直接报错或截断，导致信息丢失
- **适用场景**：知识库较小、检索到的文档数量少、总长度短的场景

### 方式 2：map_reduce

map_reduce 采用「分而治之」的策略：先把每个文档单独传给模型生成中间答案（map 阶段），再把所有中间答案汇总起来生成最终答案（reduce 阶段）。

#### 代码实现

```python
# map_reduce 方式
qa_map_reduce = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    chain_type="map_reduce"
)

# 执行问答
result = qa_map_reduce.invoke({"chat_history": memory, "question": "卢浮宫这个名字怎么来的？"})
print(result["answer"])
```

#### 优缺点分析

- **优点**：可以处理任意数量的文档，不受上下文窗口限制，每个文档都能被模型独立处理
- **缺点**：需要多次调用大模型（N 个文档至少调用 N+1 次），成本高、响应速度慢
- **适用场景**：知识库庞大、检索到的文档数量多、需要全面参考所有文档的场景

### 方式 3：refine

refine 也采用多轮调用策略，但和 map_reduce 不同：它先用第一个文档生成初始答案，然后把初始答案和第二个文档一起传给模型进行优化，得到优化后的答案，再和第三个文档一起优化，以此类推，逐步迭代 refine 最终答案。

#### 代码实现

```python
# refine 方式
qa_refine = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    chain_type="refine"
)

# 执行问答
result = qa_refine.invoke({"chat_history": memory, "question": "卢浮宫这个名字怎么来的？"})
print(result["answer"])
```

#### 优缺点分析

- **优点**：可以处理任意数量文档，且答案是通过逐步迭代优化的，可能比 map_reduce 的汇总方式质量更高
- **缺点**：同样需要多次调用模型，且优化效果依赖文档的迭代顺序，前面的文档影响更大
- **适用场景**：需要逐步精炼答案、文档之间有逻辑递进关系的复杂场景

### 方式 4：map_rerank

map_rerank 先让每个文档单独调用模型，既生成答案又给答案打分，最后选择分数最高的那个答案返回给用户。

#### 代码实现

```python
# map_rerank 方式
qa_map_rerank = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    chain_type="map_rerank"
)

# 执行问答
result = qa_map_rerank.invoke({"chat_history": memory, "question": "卢浮宫这个名字怎么来的？"})
print(result["answer"])
```

#### 优缺点分析

- **优点**：自动选出最相关文档的答案，不需要汇总多个文档，逻辑简单直接
- **缺点**：只返回一个文档的答案，可能遗漏其他文档中的重要信息，答案不够全面
- **适用场景**：只需要一个最准确答案、不需要综合多个文档信息的场景

## 四、4 种方式的快速切换演示

通过修改 chain_type 参数，可以快速在不同方式之间切换，对比它们的效果差异：

```python
# 方式 1：stuff（默认）
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    chain_type="stuff"
)

# 方式 2：map_reduce
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    chain_type="map_reduce"
)

# 方式 3：refine
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    chain_type="refine"
)

# 方式 4：map_rerank
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    chain_type="map_rerank"
)

# 执行问答（4种方式调用方式完全相同）
result = qa.invoke({"chat_history": memory, "question": "卢浮宫这个名字怎么来的？"})
print(result["answer"])
```

## 五、选择建议与最佳实践

### 选择决策树

1. **文档总长度短（不会超上下文窗口）** → 优先用 **stuff**，简单高效
2. **文档数量多、总长度长** → 考虑 **map_reduce** 或 **refine**
3. **需要综合所有文档信息生成答案** → 用 **map_reduce**
4. **文档有逻辑递进关系、需要逐步优化** → 用 **refine**
5. **只需要一个最准确的答案** → 用 **map_rerank**

### 实际建议

- 大部分场景下，**stuff 方式足够用**，先从默认方式开始测试
- 如果遇到上下文窗口溢出的错误，再考虑切换到 map_reduce 或 refine
- map_reduce 和 refine 的成本较高，生产环境需要权衡成本和效果
- 可以通过调整检索器的 top_k 参数（返回文档数量）来控制输入给模型的文档数量

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

# 3. 构建向量库和检索器
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

# 6. 创建对话链（4种方式任选其一）

# stuff 方式（默认）
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    chain_type="stuff"
)

# map_reduce 方式
# qa = ConversationalRetrievalChain.from_llm(
#     llm=model,
#     retriever=retriever,
#     memory=memory,
#     chain_type="map_reduce"
# )

# refine 方式
# qa = ConversationalRetrievalChain.from_llm(
#     llm=model,
#     retriever=retriever,
#     memory=memory,
#     chain_type="refine"
# )

# map_rerank 方式
# qa = ConversationalRetrievalChain.from_llm(
#     llm=model,
#     retriever=retriever,
#     memory=memory,
#     chain_type="map_rerank"
# )

# 7. 执行问答
result = qa.invoke({"chat_history": memory, "question": "卢浮宫这个名字怎么来的？"})
print(result["answer"])
```
