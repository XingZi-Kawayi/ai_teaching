# 5、Vector Store——向量数据库

我们的最终目标，是让 AI 能基于**我们自己的本地文档**精准回答问题，而整个流程的核心前提，就是解决「传统数据库搞不定非结构化文本语义搜索」的痛点。

- **传统数据库**：靠**精准关键词匹配**找数据，适合找员工 ID、日期这种有固定格式的结构化数据。比如你搜 "员工 ID=002" 能精准找到，但搜 "擅长做财务预算的员工"，哪怕备注里写了 "预算控制和财务规划经验丰富"，因为没有完全匹配的关键词，传统数据库根本搜不到。
- **向量数据库**：靠**语义相似性搜索**找数据，专门处理文章、新闻、文档这种没有固定格式的非结构化数据。它会把文本转换成一串数字（向量），语义越相近的文本，向量在空间里的距离就越近，哪怕关键词不完全一样，也能找到意思相关的内容。

我们接下来的所有代码操作，就是完成「把本地文档切成小块→转成向量→存入 FAISS 向量数据库→实现语义搜索」的全流程。

## 一、逐行代码 + 文档原理 全流程拆解

### 第一步：环境准备，安装核心依赖库

```bash
pip install faiss-cpu
```

这行是安装 FAISS 库。

- FAISS 是 Meta 开源的向量数据库，也是目前最常用的轻量级向量数据库之一；
- faiss-cpu是 CPU 版本，适合本地测试、小数据量场景，新手零门槛就能用；如果是海量数据，可换成 GPU 版本faiss-gpu。

### 第二步：导入所有需要的工具模块

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
```

这 4 行导入，对应「分割好的文本块处理、嵌入模型实例准备、向量数据库操作」的核心工具：

- **TextLoader**：文档加载器，专门把本地的 txt 纯文本文件读取到程序里，是所有操作的起点，没有它就拿不到要处理的文档内容；
- **FAISS**：我们要用的向量数据库本体，负责存向量、做相似性搜索；
- **OpenAIEmbeddings**：嵌入模型，核心作用是把人类能看懂的文字，转换成计算机能计算的数字向量，这是语义搜索的基础；
- **RecursiveCharacterTextSplitter**：文本分割器，负责把长文档切成合适大小的文本块，保证检索精准度。

### 第三步：加载本地待处理的文档

```python
loader = TextLoader("./demo2.txt")
docs = loader.load()
```

这两行就是把我们的目标文档读进程序里：

- 第一行：创建一个文本加载器实例，括号里./demo2.txt是你本地 txt 文件的路径，./代表当前程序运行的文件夹，你可以换成自己的文件路径；
- 第二行：调用load()方法，把 txt 文件里的全部内容，读取成程序能处理的文档对象，后续的文本分割、转向量，都基于这个读取到的内容。

### 第四步：把长文档切割成规范的文本块

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
)
texts = text_splitter.split_documents(docs)
```

这部分是「分割好的文本块处理」，也是语义检索能精准的关键：

#### 为什么要切割文档？

如果直接把一整篇几万字的文档转成向量，会出现两个问题：一是嵌入模型对输入文本长度有限制，超长文本处理不了；二是长文本转出来的向量语义很模糊，搜问题的时候找不到精准的对应内容。切成小块后，每个块的语义集中，检索更准。

#### 每个参数的含义

- **chunk_size=500**：每个文本块的最大字符数是 500，保证每个块的内容不会太长，语义足够集中；
- **chunk_overlap=40**：相邻两个文本块，会有 40 个字符的重叠内容。核心作用是防止一句话、一个完整语义被切成两半，导致信息断裂；
- **separators**：分割符的优先级，从左到右优先用前面的符号分割。先按两个换行（段落）分割，再按一个换行（行）分割，再按句号、感叹号、问号等标点分割，最后才按单个字符分割，最大程度保证不在一句话的中间切割，保留语义完整。

#### 最后一行的作用

texts = text_splitter.split_documents(docs)：用我们上面设置好的分割规则，把第一步加载进来的完整文档，切成一个个符合要求的小文本块，最终得到一个文本块列表texts，这就是后续要转成向量的素材。

### 第五步：初始化嵌入模型实例

```python
embeddings_model = OpenAIEmbeddings()
```

这一行是「嵌入模型实例的准备工作」，是连接文本和向量的核心桥梁。

- 嵌入模型的核心作用：把人类能看懂的自然语言文本，转换成一串有语义含义的数字（也就是向量 / 嵌入向量）；
- 核心逻辑：语义越相近的文本，转出来的向量在空间里的距离就越近；
- 这行代码就是实例化了 OpenAI 的嵌入模型，后续我们会用它给所有文本块、用户的查询问题转向量。

### 第六步：文本转向量，存入 FAISS 向量数据库

```python
db = FAISS.from_documents(texts, embeddings_model)
```

这一行是整个流程的核心，也是「将一个个文本块嵌入为向量并储存进向量数据库」的核心操作。

FAISS 的from_documents方法只需要传两个核心参数：

- 第一个参数：我们切好的文本块列表texts；
- 第二个参数：我们初始化好的嵌入模型实例embeddings_model；

这个方法会自动完成两件事：

1. 遍历所有文本块，用嵌入模型把每一个文本块都转换成对应的向量；
2. 把转换好的向量，和对应的原始文本块绑定，一起存入 FAISS 向量数据库，最终生成一个数据库实例db，后续所有的检索操作，都基于这个数据库实例。

### 第七步：构建检索器，为语义搜索做准备

```python
retriever = db.as_retriever()
```

这行代码的作用，就是把我们刚才建好的 FAISS 向量数据库，封装成一个 LangChain 体系里标准的检索器；

这个检索器属于 Runnable（可运行组件），后续可以直接调用它的invoke方法执行搜索，也能无缝接入后续的 AI 对话、提示词模板、记忆模块等流程。

### 第八步：执行相似性搜索，验证检索效果

```python
retrieved_docs = retriever.invoke("卢浮宫这个名字怎么来的？")
print(retrieved_docs[0].page_content)
```

这部分就是「相似性搜索实操测试」：

**核心执行逻辑**：retriever.invoke("你的问题")

- 调用检索器的invoke方法，把用户的查询问题传进去；
- 检索器会自动做两件事：先把你的查询问题，用同一个嵌入模型转换成向量；再去 FAISS 数据库里，找和查询向量距离最近、语义最相关的文本块；
- 最终返回的结果retrieved_docs，是一个文档组成的列表，而且已经**按照相似度从高到低排好序**了，排在第一个的，就是和你的问题最相关的文本块。

**打印结果验证**：print(retrieved_docs[0].page_content)

- retrieved_docs[0]：取列表里第一个、也就是相似度最高的文档；
- .page_content：取出这个文档里的文本内容，打印出来就能看到，和问题相关的答案就在这个文本块里，完美印证了「向量数据库相似性搜索的有效性」。

代码里做了两次不同的查询测试，分别对应两个不同的问题，能全面验证我们的向量数据库，能不能精准匹配到文档里不同维度的相关内容。

## 二、流程收尾与后续拓展

到上面最后一行代码为止，我们已经完整完成了两大核心步骤：**准备外部数据（文档加载→分割→转向量存库）、实现相似性搜索**。

而这只是 RAG（检索增强生成）的基础步骤：

- 想要让 AI 直接针对问题给出通顺、精准的完整回答，还需要把这里搜索出的相关文本片段，和用户的查询请求结合在一起，通过提示词模板一起传给大语言模型；
- 如果想要实现连续对话的功能，还需要在这个基础上，加上记忆相关的设置，这些进阶操作会在后续流程中完成。

## 完整代码回顾

```python
# 安装依赖
# pip install faiss-cpu

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 加载文档
loader = TextLoader("./demo2.txt")
docs = loader.load()

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
)
texts = text_splitter.split_documents(docs)

# 初始化嵌入模型
embeddings_model = OpenAIEmbeddings()

# 创建向量数据库
db = FAISS.from_documents(texts, embeddings_model)

# 构建检索器
retriever = db.as_retriever()

# 执行相似性搜索
retrieved_docs = retriever.invoke("卢浮宫这个名字怎么来的？")
print(retrieved_docs[0].page_content)
```
