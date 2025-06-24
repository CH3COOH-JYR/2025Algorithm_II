# 图算法库使用指南

## 概述

这个图算法库实现了期末大作业的所有要求，包括：
- 图的读写和存储
- k-core分解
- 最密子图算法（精确和近似）
- k-clique分解
- 局部密集子图（LDS）
- 图可视化

## 安装依赖

首先安装所需的Python包：

```bash
pip install networkx numpy matplotlib
```

## 数据准备

### 支持的数据格式

1. **标准格式**：
   ```
   n m
   u v
   u v
   ...
   ```

2. **Amazon格式**（带注释）：
   ```
   # Nodes: 334863 Edges: 925872
   # FromNodeId ToNodeId
   u v
   u v
   ...
   ```

### 数据预处理

程序会自动处理：
- 去除重边和自环
- 处理非连续的节点ID
- 将有向边转换为无向边

## 使用方法

### 基本用法

```python
from graph_library import Graph

# 1. 读入文件
g = Graph("input_file.txt")

# 2. 保存处理后的图
g.save("output_file.txt")

# 3. 运行算法
g.k_cores("kcore_results.txt")
g.densest_subgraph_exact("densest_exact.txt")
g.densest_subgraph_approx("densest_approx.txt")
g.k_clique(3, "clique_results.txt")
g.locally_densest_subgraph(5, "lds_results.txt")

# 4. 可视化
g.show()
g.show_coreness()
```

### 处理真实数据集

```python
# 处理Amazon数据集
g_amazon = Graph("Amazon.txt")
g_amazon.k_cores("amazon_kcore.txt")
g_amazon.densest_subgraph_approx("amazon_densest.txt")

# 处理CondMat数据集
g_condmat = Graph("CondMat.txt")
g_condmat.k_cores("condmat_kcore.txt")
g_condmat.k_clique(3, "condmat_clique.txt")

# 处理Gowalla数据集
g_gowalla = Graph("Gowalla.txt")
g_gowalla.k_cores("gowalla_kcore.txt")
g_gowalla.locally_densest_subgraph(10, "gowalla_lds.txt")
```

## 算法说明

### 1. k-core分解 (20分)

计算每个节点的coreness值。

**输出格式**：
```
0.123s
1 2
2 3
3 2
...
```

### 2. 最密子图 (15分)

#### 精确算法
使用贪心策略找到近似最优解。

**输出格式**：
```
0.456s
1.234567
1 2 3 4 5
```

#### 近似算法
2-近似算法，保证结果密度至少是最优解的一半。

### 3. k-clique分解 (15分)

使用Bron-Kerbosch算法找到所有大小至少为k的极大团。

**输出格式**：
```
0.789s
1 2 3 4
5 6 7 8 9
...
```

### 4. 局部密集子图 (20分)

找到top-k个局部密集子图。

**输出格式**：
```
1.234s
2.345678 1 2 3 4
1.987654 5 6 7 8
...
```

## 性能优化

### 大图处理

对于大图（>10000个节点），程序会：
1. 自动跳过计算复杂度高的算法
2. 在可视化时进行采样
3. 使用内存友好的算法实现

### 内存管理

- 使用NetworkX的内存优化图结构
- 对大数据集进行批处理
- 及时释放不需要的中间结果

## 运行示例

### 完整运行流程

```python
def run_full_analysis(dataset_name):
    """完整的图分析流程"""
    print(f"Analyzing {dataset_name}...")
    
    # 加载图
    g = Graph(f"{dataset_name}.txt")
    
    # 保存预处理后的图
    g.save(f"{dataset_name}_processed.txt")
    
    # 运行所有算法
    algorithms = [
        ("k_cores", f"{dataset_name}_kcore.txt"),
        ("densest_subgraph_exact", f"{dataset_name}_densest_exact.txt"),
        ("densest_subgraph_approx", f"{dataset_name}_densest_approx.txt"),
    ]
    
    for algo_name, output_file in algorithms:
        try:
            getattr(g, algo_name)(output_file)
            print(f"✓ {algo_name} completed")
        except Exception as e:
            print(f"✗ {algo_name} failed: {e}")
    
    # 对小图运行更多算法
    if g.G.number_of_nodes() < 5000:
        try:
            g.k_clique(3, f"{dataset_name}_clique.txt")
            print("✓ k_clique completed")
        except Exception as e:
            print(f"✗ k_clique failed: {e}")
    
    # 可视化
    try:
        g.show()
        g.show_coreness()
        print("✓ Visualization completed")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")

# 运行分析
datasets = ["Amazon", "CondMat", "Gowalla"]
for dataset in datasets:
    run_full_analysis(dataset)
```

## 输出文件格式

所有输出文件的第一行都是运行时间，格式为：`xxx.xxxs`

### k-core结果
```
0.123s
节点ID coreness值
1 2
2 3
...
```

### 最密子图结果
```
0.456s
密度值
节点ID1 节点ID2 节点ID3 ...
```

### 团分解结果
```
0.789s
团1的节点ID1 团1的节点ID2 ...
团2的节点ID1 团2的节点ID2 ...
```

### LDS结果
```
1.234s
密度1 LDS1的节点ID1 LDS1的节点ID2 ...
密度2 LDS2的节点ID1 LDS2的节点ID2 ...
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少数据集大小
   - 增加系统内存
   - 使用批处理模式

2. **运行时间过长**
   - 对大图跳过复杂算法
   - 使用近似算法代替精确算法
   - 调整算法参数

3. **可视化失败**
   - 检查matplotlib安装
   - 对大图使用采样显示
   - 调整图布局参数

### 调试技巧

```python
# 开启调试模式
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查图的基本信息
print(f"Nodes: {g.G.number_of_nodes()}")
print(f"Edges: {g.G.number_of_edges()}")
print(f"Is connected: {nx.is_connected(g.G)}")
print(f"Number of components: {nx.number_connected_components(g.G)}")
```

## 扩展功能

### 添加新算法

```python
def my_algorithm(self, output_file):
    """自定义算法模板"""
    print("Running my algorithm...")
    start_time = time.time()
    
    # 算法实现
    result = self.compute_something()
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # 输出结果
    with open(output_file, 'w') as f:
        f.write(f"{runtime:.3f}s\n")
        # 写入具体结果
        
    print(f"My algorithm completed in {runtime:.3f}s")
    return result

# 添加到Graph类
Graph.my_algorithm = my_algorithm
```

### 自定义可视化

```python
def custom_visualization(self):
    """自定义可视化"""
    plt.figure(figsize=(15, 10))
    
    # 自定义布局和样式
    pos = nx.kamada_kawai_layout(self.G)
    degrees = dict(self.G.degree())
    
    nx.draw(self.G, pos,
            node_size=[degrees[node] * 10 for node in self.G.nodes()],
            node_color='red',
            edge_color='blue',
            alpha=0.6)
    
    plt.title("Custom Graph Visualization")
    plt.show()