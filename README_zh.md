# ⚡ CompactBM25

> **[English](README.md) | 中文**

> **内存高效、API 风格与 `rank_bm25` 相近的 BM25Okapi 实现。**
> 基于 NumPy CSR/CSC 稀疏矩阵，将 RAG 索引内存砍掉 **60% ~ 80%**，查询速度提升 **10x~20x**。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 解决什么问题

在现代 LLM RAG 管道中，**混合检索（向量 + BM25）** 是保证召回精度的黄金标准，几乎人人都在用 `rank_bm25.BM25Okapi`。

但当知识库规模超过 10 万文档块（chunks）时：
- **内存爆炸**：`rank_bm25` 用 Python `list[dict]` 存储词频，生产环境轻松吃掉 **1.5 GB ~ 3 GB 内存**。
- **冷启动痛苦**：pickle 反序列化百万级嵌套字典，容器启动要 **70 秒以上**，还经常引发内存尖峰导致容器 OOM 被杀。

---

## 💡 方案：CompactBM25

**CompactBM25** 用连续的 **NumPy 扁平数组** 和 **CSR（压缩稀疏行）/ CSC（压缩稀疏列）** 偏移量彻底重构了 BM25 的存储方式：

- 📉 **内存直降 60% ~ 80%**：紧凑 CSR 表示，把内存占用从 GB 级压到几百 MB。
- 🎯 **数值兼容**：复刻 `rank_bm25.BM25Okapi` 的 Robertson IDF 公式与 epsilon 地板逻辑，并在测试浮点容差内保持一致。
- ⚡ **紧凑序列化与快速查询**：用扁平 NumPy 数组替代嵌套词频字典，并支持向量化打分。
- 🔄 **熟悉的 API**：支持 `bm25.get_scores(query)` 和 `bm25.get_top_n(query, docs)`，核心行为配有一致性测试。

---

## 📊 性能基准

随附脚本 `benchmarks/benchmark_vs_rank_bm25.py` 的一次本地运行结果如下（5,000 篇合成文档、每篇 50 词、词表大小 2,000）：

| 指标 | `rank_bm25.BM25Okapi` | `CompactBM25` | 提升 |
| :--- | :--- | :--- | :--- |
| **峰值内存** | 11.08 MB | **3.30 MB** | **📉 节省 70.2%** |
| **查询延迟（100 次均值）** | 1.45 ms | **0.05 ms** | **⚡ 快 29 倍** |
| **最大评分差异** | 基准 | **0.00e+00** | **✅ 本次运行数值相等** |

结果会受到 Python、NumPy、硬件、语料结构和词表分布影响；容量规划前请在自己的环境中运行随附基准测试。

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/wangkun5212-hue/compact-bm25.git
cd compact-bm25
pip install .
```

### 基础用法（零改造替换）

```python
from compact_bm25 import CompactBM25

corpus = [
    ["deep", "learning", "model"],
    ["natural", "language", "processing"],
    ["deep", "neural", "network"],
    ["retrieval", "augmented", "generation"]
]

# 1. 从分词后的语料构建索引
bm25 = CompactBM25.build(corpus)

# 2. 对查询打分
query = ["deep", "learning"]
scores = bm25.get_scores(query)
print("BM25 Scores:", scores)

# 3. 直接取 Top-N 文档
docs = ["Doc A", "Doc B", "Doc C", "Doc D"]
top_docs = bm25.get_top_n(query, docs, n=2)
print("Top 2 Matches:", top_docs)
```

### 持久化（闪电级序列化）

由于内部缓冲全部是扁平连续的 NumPy 数组，序列化极其简单且体积极小：

```python
import pickle

# 保存到磁盘（pickle 体积比 rank_bm25 小约 10 倍）
with open("bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

# 重新加载紧凑索引
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)
```

---

## 🔬 实现原理

**CompactBM25** 不再存储 Python 对象树（`dict[term -> count]`），而是把文档与词项倒排记录打包进平行扁平数组：
1. `offsets [int64]`：CSR 行指针，指向文档边界。
2. `term_ids [int32]`：整数词项 ID 的扁平数组。
3. `freqs [int8/int16/int32]`：自适应向下转型（downcast）的整数词频。
4. `term_offsets & inv_doc [CSC]`：倒排记录表，查询时支持 NumPy 向量化切片累加。

---

## 🧪 运行测试与基准

```bash
git clone https://github.com/wangkun5212-hue/compact-bm25.git
cd compact-bm25
pip install -r requirements.txt

# 跑一致性校验测试
pytest tests/ -v

# 跑性能与内存基准对比
python benchmarks/benchmark_vs_rank_bm25.py
```

---

## 📄 许可证

MIT License。商业与研究用途均可免费使用。
