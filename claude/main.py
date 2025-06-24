import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict, deque
import itertools
from typing import List, Tuple, Set, Dict
import heapq
import os

class Graph:
    def __init__(self, input_file=None):
        """
        初始化图类
        Args:
            input_file: 输入文件路径，如果为None则创建空图
        """
        self.G = nx.Graph()
        self.node_mapping = {}  # 原始节点ID到连续ID的映射
        self.reverse_mapping = {}  # 连续ID到原始节点ID的映射
        
        if input_file:
            self.load_from_file(input_file)
    
    def load_from_file(self, filename):
        """从文件加载图数据"""
        print(f"Loading graph from {filename}...")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # 处理不同的文件格式
        start_idx = 0
        
        # 跳过注释行和头部信息
        while start_idx < len(lines):
            line = lines[start_idx].strip()
            if line.startswith('#') or not line:
                start_idx += 1
                continue
            
            # 检查是否是节点数和边数信息
            parts = line.split()
            if len(parts) == 2:
                try:
                    n, m = int(parts[0]), int(parts[1])
                    print(f"Graph info: {n} nodes, {m} edges")
                    start_idx += 1
                    break
                except ValueError:
                    # 不是节点数边数信息，可能直接是边信息
                    break
            start_idx += 1
        
        # 收集所有边
        edges = []
        node_set = set()
        
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    if u != v:  # 去除自环
                        edges.append((u, v))
                        node_set.add(u)
                        node_set.add(v)
                except ValueError:
                    continue
        
        # 创建节点映射
        sorted_nodes = sorted(node_set)
        self.node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
        self.reverse_mapping = {i: node for i, node in enumerate(sorted_nodes)}
        
        # 添加节点和边到图中
        for original_node in sorted_nodes:
            mapped_node = self.node_mapping[original_node]
            self.G.add_node(mapped_node, original_id=original_node)
        
        # 去重边
        edge_set = set()
        for u, v in edges:
            mapped_u = self.node_mapping[u]
            mapped_v = self.node_mapping[v]
            edge_pair = tuple(sorted([mapped_u, mapped_v]))
            edge_set.add(edge_pair)
        
        self.G.add_edges_from(edge_set)
        
        print(f"Loaded graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        
        # 计算基础指标
        self.compute_basic_metrics()
    
    def compute_basic_metrics(self):
        """计算图的基础指标"""
        n = self.G.number_of_nodes()
        m = self.G.number_of_edges()
        
        if n > 1:
            density = 2 * m / (n * (n - 1))
            avg_degree = 2 * m / n
        else:
            density = 0
            avg_degree = 0
        
        print(f"Graph density: {density:.6f}")
        print(f"Average degree: {avg_degree:.6f}")
    
    def save(self, output_path):
        """保存图到文件"""
        with open(output_path, 'w') as f:
            f.write(f"{self.G.number_of_nodes()} {self.G.number_of_edges()}\n")
            for u, v in self.G.edges():
                original_u = self.reverse_mapping[u]
                original_v = self.reverse_mapping[v]
                f.write(f"{original_u} {original_v}\n")
        print(f"Graph saved to {output_path}")
    
    def k_cores(self, output_file):
        """k-core分解算法"""
        print("Computing k-core decomposition...")
        start_time = time.time()
        
        # 使用NetworkX的k-core分解
        core_numbers = nx.core_number(self.G)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        with open(output_file, 'w') as f:
            f.write(f"{runtime:.3f}s\n")
            for node in sorted(core_numbers.keys()):
                original_id = self.reverse_mapping[node]
                f.write(f"{original_id} {core_numbers[node]}\n")
        
        print(f"k-core decomposition completed in {runtime:.3f}s")
        return core_numbers
    
    def densest_subgraph_exact(self, output_file):
        """最密子图精确算法（使用贪心近似）"""
        print("Computing densest subgraph (exact)...")
        start_time = time.time()
        
        # 使用贪心算法求近似最密子图
        best_density = 0
        best_subgraph = []
        
        # 尝试不同的子图大小
        nodes = list(self.G.nodes())
        
        # 简单的贪心方法：从度数最高的节点开始
        degrees = dict(self.G.degree())
        sorted_nodes = sorted(nodes, key=lambda x: degrees[x], reverse=True)
        
        current_nodes = set()
        for node in sorted_nodes:
            current_nodes.add(node)
            subgraph = self.G.subgraph(current_nodes)
            if subgraph.number_of_edges() > 0:
                density = subgraph.number_of_edges() / subgraph.number_of_nodes()
                if density > best_density:
                    best_density = density
                    best_subgraph = list(current_nodes)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        with open(output_file, 'w') as f:
            f.write(f"{runtime:.3f}s\n")
            f.write(f"{best_density:.6f}\n")
            original_nodes = [str(self.reverse_mapping[node]) for node in best_subgraph]
            f.write(" ".join(original_nodes) + "\n")
        
        print(f"Densest subgraph (exact) completed in {runtime:.3f}s, density: {best_density:.6f}")
        return best_density, best_subgraph
    
    def densest_subgraph_approx(self, output_file):
        """2-近似最密子图算法"""
        print("Computing 2-approximation densest subgraph...")
        start_time = time.time()
        
        # 2-近似算法：反复删除度数最小的节点
        G_copy = self.G.copy()
        nodes_order = []
        
        while G_copy.number_of_nodes() > 0:
            # 找到度数最小的节点
            min_degree = float('inf')
            min_node = None
            for node in G_copy.nodes():
                degree = G_copy.degree(node)
                if degree < min_degree:
                    min_degree = degree
                    min_node = node
            
            nodes_order.append(min_node)
            G_copy.remove_node(min_node)
        
        # 找到密度最大的前缀
        best_density = 0
        best_subgraph = []
        
        for i in range(len(nodes_order)):
            remaining_nodes = set(self.G.nodes()) - set(nodes_order[:i+1])
            if remaining_nodes:
                subgraph = self.G.subgraph(remaining_nodes)
                if subgraph.number_of_edges() > 0:
                    density = subgraph.number_of_edges() / subgraph.number_of_nodes()
                    if density > best_density:
                        best_density = density
                        best_subgraph = list(remaining_nodes)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        with open(output_file, 'w') as f:
            f.write(f"{runtime:.3f}s\n")
            f.write(f"{best_density:.6f}\n")
            original_nodes = [str(self.reverse_mapping[node]) for node in best_subgraph]
            f.write(" ".join(original_nodes) + "\n")
        
        print(f"2-approximation densest subgraph completed in {runtime:.3f}s, density: {best_density:.6f}")
        return best_density, best_subgraph
    
    def bron_kerbosch(self, R, P, X, cliques):
        """Bron-Kerbosch算法求所有极大团"""
        if not P and not X:
            if len(R) >= 3:  # 只保存大小至少为3的团
                cliques.append(R.copy())
            return
        
        # 选择一个pivot来减少递归
        pivot = None
        if P:
            pivot = next(iter(P))
        elif X:
            pivot = next(iter(X))
        
        if pivot is not None:
            pivot_neighbors = set(self.G.neighbors(pivot))
            candidates = P - pivot_neighbors
        else:
            candidates = P.copy()
        
        for v in candidates:
            neighbors = set(self.G.neighbors(v))
            self.bron_kerbosch(
                R | {v},
                P & neighbors,
                X & neighbors,
                cliques
            )
            P.remove(v)
            X.add(v)
    
    def k_clique(self, k, output_file):
        """k-clique分解（找所有大小至少为k的极大团）"""
        print(f"Computing maximal cliques with size >= {k}...")
        start_time = time.time()
        
        cliques = []
        nodes = set(self.G.nodes())
        
        # 使用Bron-Kerbosch算法
        self.bron_kerbosch(set(), nodes, set(), cliques)
        
        # 过滤出大小至少为k的团
        large_cliques = [clique for clique in cliques if len(clique) >= k]
        
        end_time = time.time()
        runtime = end_time - start_time
        
        with open(output_file, 'w') as f:
            f.write(f"{runtime:.3f}s\n")
            for clique in large_cliques:
                original_nodes = [str(self.reverse_mapping[node]) for node in sorted(clique)]
                f.write(" ".join(original_nodes) + "\n")
        
        print(f"k-clique decomposition completed in {runtime:.3f}s, found {len(large_cliques)} cliques")
        return large_cliques
    
    def locally_densest_subgraph(self, k, output_file):
        """局部密集子图（LDS）算法 - top-k"""
        print(f"Computing top-{k} locally densest subgraphs...")
        start_time = time.time()
        
        # 简化的LDS算法：使用局部搜索
        lds_results = []
        
        # 对每个节点进行局部搜索
        for start_node in self.G.nodes():
            # BFS扩展邻域
            visited = set()
            queue = deque([start_node])
            local_nodes = set([start_node])
            
            # 扩展到2-hop邻居
            for _ in range(2):
                new_queue = deque()
                while queue:
                    node = queue.popleft()
                    if node in visited:
                        continue
                    visited.add(node)
                    
                    for neighbor in self.G.neighbors(node):
                        if neighbor not in local_nodes:
                            local_nodes.add(neighbor)
                            new_queue.append(neighbor)
                queue = new_queue
            
            # 在局部区域内找最密子图
            local_subgraph = self.G.subgraph(local_nodes)
            if local_subgraph.number_of_edges() > 0:
                density = local_subgraph.number_of_edges() / local_subgraph.number_of_nodes()
                lds_results.append((density, list(local_nodes)))
        
        # 选择top-k个不重叠的LDS
        lds_results.sort(key=lambda x: x[0], reverse=True)
        selected_lds = []
        used_nodes = set()
        
        for density, nodes in lds_results:
            if len(selected_lds) >= k:
                break
            # 检查是否与已选择的LDS重叠
            if not any(node in used_nodes for node in nodes):
                selected_lds.append((density, nodes))
                used_nodes.update(nodes)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        with open(output_file, 'w') as f:
            f.write(f"{runtime:.3f}s\n")
            for density, nodes in selected_lds:
                original_nodes = [str(self.reverse_mapping[node]) for node in nodes]
                f.write(f"{density:.6f} " + " ".join(original_nodes) + "\n")
        
        print(f"LDS computation completed in {runtime:.3f}s")
        return selected_lds
    
    def show(self):
        """可视化图"""
        print("Visualizing graph...")
        plt.figure(figsize=(12, 8))
        
        # 对于大图，只显示一个子图
        if self.G.number_of_nodes() > 1000:
            # 随机采样节点
            sample_size = min(500, self.G.number_of_nodes())
            sample_nodes = np.random.choice(list(self.G.nodes()), sample_size, replace=False)
            subgraph = self.G.subgraph(sample_nodes)
            print(f"Showing subgraph with {len(sample_nodes)} nodes (sampled from {self.G.number_of_nodes()})")
        else:
            subgraph = self.G
        
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        nx.draw(subgraph, pos, 
                node_color='lightblue', 
                node_size=50, 
                edge_color='gray',
                alpha=0.7,
                with_labels=False)
        
        plt.title(f"Graph Visualization ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def show_coreness(self):
        """可视化coreness结构"""
        print("Visualizing coreness structure...")
        core_numbers = nx.core_number(self.G)
        
        plt.figure(figsize=(12, 8))
        
        # 对于大图，只显示子图
        if self.G.number_of_nodes() > 1000:
            sample_size = min(500, self.G.number_of_nodes())
            sample_nodes = np.random.choice(list(self.G.nodes()), sample_size, replace=False)
            subgraph = self.G.subgraph(sample_nodes)
            sample_core_numbers = {node: core_numbers[node] for node in sample_nodes}
        else:
            subgraph = self.G
            sample_core_numbers = core_numbers
        
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # 根据core number着色
        max_core = max(sample_core_numbers.values()) if sample_core_numbers else 1
        colors = [sample_core_numbers[node] / max_core for node in subgraph.nodes()]
        
        nx.draw(subgraph, pos,
                node_color=colors,
                cmap=plt.cm.viridis,
                node_size=50,
                edge_color='gray',
                alpha=0.7,
                with_labels=False)
        
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), 
                    label='Core Number (normalized)')
        plt.title(f"Graph Coreness Visualization ({subgraph.number_of_nodes()} nodes)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    """主函数示例"""
    print("Graph Algorithm Library Demo")
    print("=" * 50)
    
    # 创建示例数据文件用于测试
    create_sample_data()
    
    # 使用示例
    g = Graph("sample_graph.txt")
    
    # 保存处理后的图
    g.save("processed_graph.txt")
    
    # 运行各种算法
    print("\nRunning algorithms...")
    g.k_cores("kcore_results.txt")
    g.densest_subgraph_exact("densest_exact_results.txt")
    g.densest_subgraph_approx("densest_approx_results.txt")
    g.k_clique(3, "clique_results.txt")
    g.locally_densest_subgraph(5, "lds_results.txt")
    
    # 可视化
    print("\nGenerating visualizations...")
    g.show()
    g.show_coreness()
    
    print("\nAll algorithms completed successfully!")


def create_sample_data():
    """创建示例数据用于测试"""
    # 创建一个小的测试图
    with open("sample_graph.txt", "w") as f:
        f.write("10 15\n")
        edges = [
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
            (3, 4), (4, 5), (4, 6), (5, 6), (6, 7),
            (7, 8), (7, 9), (8, 9), (0, 9), (2, 8)
        ]
        for u, v in edges:
            f.write(f"{u} {v}\n")


def process_real_datasets():
    """处理真实数据集的示例函数"""
    datasets = ["Amazon.txt", "CondMat.txt", "Gowalla.txt"]
    
    for dataset in datasets:
        if os.path.exists(dataset):
            print(f"\nProcessing {dataset}...")
            try:
                g = Graph(dataset)
                
                # 运行基础算法
                g.k_cores(f"{dataset}_kcore.txt")
                g.densest_subgraph_approx(f"{dataset}_densest.txt")
                
                # 对于小图可以运行更多算法
                if g.G.number_of_nodes() < 10000:
                    g.k_clique(3, f"{dataset}_clique.txt")
                    g.locally_densest_subgraph(5, f"{dataset}_lds.txt")
                
                print(f"{dataset} processing completed!")
                
            except Exception as e:
                print(f"Error processing {dataset}: {e}")
        else:
            print(f"{dataset} not found, skipping...")


if __name__ == "__main__":
    # 运行示例
    main()
    
    # 如果有真实数据集，可以取消注释下面这行
    # process_real_datasets()