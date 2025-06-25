# 算法课期末大作业 - 图结构挖掘系统

## 项目概述

本项目实现了一个完整的图结构挖掘系统，包含图的读写、结构挖掘算法实现以及图可视化功能。

## 功能特性

### 1. 图的读写 (20分)
- ✅ 支持多种图文件格式的读取
- ✅ 自动处理重边和自环
- ✅ 节点ID映射和重新编号
- ✅ 图的基础指标计算（密度、平均度等）
- ✅ 图的保存功能

### 2. 图结构挖掘算法 (60分)
- ✅ **k-core分解** (10分): 计算每个顶点的coreness值
- ✅ **最密子图** (15分): 
  - 精确算法：求解最密密度子图
  - 近似算法：2-近似算法
- ✅ **k-clique分解** (15分): 使用Bron-Kerbosch算法求极大团
- ✅ **k-clique最密子图** (20分): 实现k-clique最密子图算法

### 3. 图可视化 (20分)
- ✅ 基础图可视化
- ✅ k-core结构可视化
- ✅ 支持大图的子采样可视化
- ✅ 可自定义样式和布局

## 环境要求

- Python 3.7+
- NetworkX >= 3.0
- Matplotlib >= 3.5.0
- NumPy >= 1.21.0

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据集

### 原始数据集
项目包含以下三个真实网络数据集：
- **Amazon.txt**: 334,863个节点，925,872条边 (商品共购买网络)
- **CondMat.txt**: 23,133个节点，93,497条边 (凝聚态物理合作网络)
- **Gowalla.txt**: 196,591个节点，950,327条边 (位置社交网络)

### 测试数据集
系统会自动生成以下测试数据集，适合算法演示和验证：
- **scale_free_500.txt**: 500个节点的无标度网络，适合所有算法
- **community_300.txt**: 300个节点的社区结构图，适合k-core分析
- **dense_regions_350.txt**: 350个节点的密集区域图，适合最密子图算法
- **multi_clique_80.txt**: 85个节点的多团结构图，适合极大团算法
- **small_world_400.txt**: 400个节点的小世界网络
- **large_scale_free_1000.txt**: 1000个节点的大规模测试

## 使用方法

### 1. 快速开始 (推荐)

```bash
python run_demo.py
```

这是一个交互式启动脚本，会自动检查环境、生成测试数据，并提供多种运行模式选择。

### 2. 交互式可视化界面 (主要功能)

```python
python interactive_visualization.py
```

启动带GUI的交互式可视化系统，具有以下功能：
- 📁 **文件操作**: 加载图文件或选择测试数据
- 🔬 **算法执行**: 一键运行k-core分解、最密子图、极大团检测
- 🎨 **可视化选项**: 
  - 多种布局算法 (spring, circular, random, kamada_kawai)
  - 多种视图模式 (原始图、k-core着色、最密子图高亮、极大团着色)
- ⚙️ **参数调节**: 节点大小、边宽度等可视化参数
- 📊 **实时信息**: 图的统计信息和算法结果显示
- 🔍 **交互操作**: 缩放、平移、工具栏功能

### 3. 生成测试数据

```python
python generate_test_data.py
```

生成多种类型的测试图:
- 无标度网络 (scale-free network)
- 小世界网络 (small-world network)  
- 社区结构图 (community structure)
- 密集区域图 (dense regions)
- 多团结构图 (multiple cliques)

### 4. 命令行批处理模式

```python
python graph_system_no_gui.py
```

无GUI版本，自动处理所有数据集并生成结果文件。

### 5. 小图测试

```python
python test_small_graph.py
```

创建简单测试图并验证所有算法功能。

### 6. 单独使用Graph类

```python
from graph_system import Graph

# 读入文件
g = Graph("输入文件.txt")

# 保存图
g.save("输出路径.txt")

# 实现图结构挖掘算法
g.k_cores("kcores结果.txt")
g.densest_subgraph_exact("最密子图结果.txt")
g.densest_subgraph_approx("近似最密子图结果.txt")
g.maximal_cliques("极大团结果.txt")
g.k_clique_densest(3, "k-clique结果.txt")

# 可视化
g.show()  # 展示图
g.show_coreness()  # 展示coreness结构
```

## 输出格式

### k-core分解结果
```
运行时间(秒)
节点ID coreness值
...
```

### 最密子图结果
```
运行时间(秒)
密度值
节点ID列表
```

### 极大团结果
```
运行时间(秒)
极大团1的节点列表
极大团2的节点列表
...
```

### k-clique最密子图结果
```
运行时间(秒)
密度值 节点ID列表
```

## 文件结构

```
├── graph_system.py          # 主要图系统实现
├── test_small_graph.py      # 测试脚本
├── requirements.txt         # Python依赖
├── README.md               # 说明文档
├── Amazon.txt              # Amazon数据集
├── CondMat.txt             # CondMat数据集
├── Gowalla.txt             # Gowalla数据集
└── output/                 # 输出结果目录
    ├── *_kcores.txt
    ├── *_densest_exact.txt
    ├── *_densest_approx.txt
    ├── *_maximal_cliques.txt
    └── *_kclique_densest.txt
```

## 算法实现说明

### k-core分解
- 使用NetworkX的高效实现
- 时间复杂度: O(m + n)
- 基于剥离算法(peeling algorithm)

### 最密子图算法
- **精确算法**: 使用NetworkX的densest_subgraph实现
- **近似算法**: 贪心剥离算法，2-近似保证
- 时间复杂度: O(m + n)

### k-clique分解 (极大团)
- 使用Bron-Kerbosch算法
- NetworkX的find_cliques实现
- 支持所有极大团的枚举

### k-clique最密子图
- 基于极大团检测
- 计算所有k-clique的密度
- 返回密度最高的k-clique

## 性能优化

1. **大图处理**: 对于大图自动跳过计算复杂度高的算法
2. **可视化优化**: 大图使用子采样可视化
3. **内存管理**: 使用生成器避免存储所有中间结果
4. **算法选择**: 根据图大小选择合适的算法实现

## 注意事项

1. 极大团算法对于大图(>10,000节点)会自动跳过，因为计算复杂度较高
2. 可视化功能对于大图使用采样显示
3. 所有输出文件都包含运行时间信息
4. 节点ID会自动映射到连续整数，但输出时会还原为原始ID

## 作者

算法课期末大作业 - 图结构挖掘系统