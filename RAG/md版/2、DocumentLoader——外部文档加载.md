# 2、DocumentLoader——外部文档加载（加载）

在 RAG（检索增强生成）的完整流程里，**加载**是第一步，也是后续分割、嵌入等步骤的基础，这一步需要用到 Langchain 的DocumentLoader（文档加载器）组件，它的核心作用就是把不同来源、不同格式的内容，统一加载到程序里供后续处理。

Langchain Community 的document_loaders模块下有上百种加载器，适配各种场景，接下来我们从**本地纯文本 TXT**、**本地 PDF**、**网络维基百科内容**这三个最常用的场景入手，结合代码一步步讲清楚怎么用，所有代码核心逻辑不变，同时拆解每一步的作用和返回结果。

## 一、加载本地 TXT 纯文本文件

纯文本是没有粗体、下划线、字号等格式的文本，TXT 是最典型的代表，用TextLoader加载纯文本能减少格式干扰、节省文件体积，也是最简单的加载方式，**无需额外安装依赖**。

### 核心代码 + 逐行解析

```python
from langchain_community.document_loaders import TextLoader

# 创建TextLoader实例，指定要加载的文件路径
loader = TextLoader("./demo.txt")

# 执行加载，返回Document实例的列表
docs = loader.load()

# 查看第一个文档的内容
print(docs[0].page_content)
```

**导入模块**：从 Langchain 社区库中拿出专门处理纯文本的TextLoader，这是加载 TXT 的 "专属工具"；

**创建实例**：告诉加载器要加载的文件在哪里，./demo.txt表示和代码文件同目录下的 demo.txt 文件；

**执行加载**：load()是所有加载器的通用执行方法，调用后才会真正读取文件内容；

**关键返回结果**：
- docs是一个**列表**，列表里的每个元素都是Document类的实例；
- 每个Document实例有两个核心属性：
  - page_content：字符串类型，存的是文件里的实际文本内容，用print(docs[0].page_content)就能直接查看；
  - metadata：字典类型，存的是文本的元数据（比如文件名称、文件路径、编码格式等），是程序自动记录的辅助信息。

## 二、加载本地 PDF 文件

PDF 是办公和学术中最常用的格式，比 TXT 复杂（包含页码、排版、图片等），因此PyPDFLoader（PDF 加载器）需要基于专门的 PDF 处理 Python 库**pypdf**工作，所以第一步要先安装这个依赖。

### 安装依赖

```bash
pip install pypdf
```

### 核心代码 + 逐行解析

```python
from langchain_community.document_loaders import PyPDFLoader

# 创建PyPDFLoader实例，指定要加载的PDF文件路径
loader = PyPDFLoader("./demo.pdf")

# 执行加载，返回Document实例的列表
docs = loader.load()

# 查看第一个文档的内容
print(docs[0].page_content)
```

**安装依赖**：用pip install pypdf安装专门处理 PDF 的库，安装后后续使用无需重复执行；

**导入模块**：拿出 Langchain 中基于 pypdf 的PyPDFLoader，是加载 PDF 的 "专属工具"；

**创建实例**：传入文件路径；

**执行加载**：调用load()执行加载；

**关键返回结果**：
- 同样返回**Document 实例的列表**，列表的长度通常和 PDF 的页数相关（一页对应一个 Document 实例）；
- page_content是 PDF 每页的文本内容（会自动提取纯文本，忽略图片 / 复杂排版）；
- metadata里会额外记录**PDF 的页码**，方便后续定位内容来源。

## 三、加载网络内容：维基百科词条

Langchain 的加载器不仅能加载本地文件，还能直接加载网络内容，维基百科是最常用的百科类数据源，用WikipediaLoader加载，需要先安装专门的维基百科访问依赖**wikipedia**，参数比本地文件多一点，可定制化加载需求。

### 安装依赖

```bash
pip install wikipedia
```

### 核心代码 + 逐行解析

```python
from langchain_community.document_loaders import WikipediaLoader

# 创建WikipediaLoader实例，传入查询参数
loader = WikipediaLoader(
    query="颐和园",
    lang="zh",
    load_max_docs=3
)

# 执行加载，返回Document实例的列表
docs = loader.load()

# 查看第一个文档的内容
print(docs[0].page_content)
```

**安装依赖**：pip install wikipedia让程序能访问维基百科的接口，仅首次安装即可；

**导入模块**：拿出专门加载维基百科内容的WikipediaLoader；

**创建实例**（核心区别：不传文件路径，传定制化参数）：
- query：必填参数，指定要加载的维基百科**词条名**（比如这里的 "颐和园"）；
- lang：可选参数，指定词条语言，zh表示中文，默认是英文，按需设置即可；
- load_max_docs：可选参数，限制最大加载的内容条数，避免词条内容过长一次性加载过多，这里设置 3 表示最多加载 3 个 Document 实例的内容；

**执行加载**：调用load()执行加载，返回**Document 实例的列表**；

**关键返回结果**：
- 列表长度**不超过**load_max_docs设置的数值；
- page_content是维基百科词条的文本内容，metadata会记录词条名、语言、来源链接等信息。

## 四、更多支持的加载器和查询方式

以上三个是最常用的场景，而 Langchain 的document_loaders模块的能力远不止于此，几乎覆盖了所有常见的文件格式和网络内容来源：

**本地文档格式**：JSON、CSV、Word（docx）、PPT（pptx）、Excel 等；

**互联网内容**：普通网页 URL、YouTube 视频字幕、GitHub 仓库 / 文件、Notion、飞书文档等。

所有支持的加载器类型，都可以去**Langchain 官方文档**的「Langchain Community → document loaders」板块查看，每个加载器都有对应的使用示例，核心使用逻辑和我们讲的三种一致：**安装依赖（如需）→ 导入加载器 → 创建实例传参数 → 调用 load () 加载**。

## 五、所有加载器的通用核心总结

不管加载哪种格式、哪个来源的内容，Langchain 文档加载器的**通用流程**完全一致，这也是 RAG 加载步骤的核心逻辑：

1. **匹配场景**：根据内容格式 / 来源，选择对应的专属加载器；
2. **环境准备**：安装该加载器所需的第三方依赖（纯文本 TXT 等简单格式无需）；
3. **实例化**：创建加载器对象，传入核心参数（本地文件传路径、网络内容传查询 / 链接等）；
4. **执行加载**：调用load()方法，得到**Document 实例的列表**；
5. **后续衔接**：这个列表就是 RAG 下一步**分割**的输入，所有后续操作都基于这个统一的格式展开。

简单来说，文档加载器的作用就是**把 "五花八门的内容" 变成 "程序能统一处理的 Document 列表"**，是 RAG 流程中实现 "内容标准化" 的第一步。
