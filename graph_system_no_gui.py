import networkx as nx
import numpy as np
import time
from collections import defaultdict, deque
from itertools import combinations
import heapq
from typing import List, Set, Tuple, Dict, Optional
import pickle

class Graph:
    """
    图结构挖掘系统 - 算法课期末大作业 (无GUI版本)
    
    功能包括：
    1. 图的读写和基础操作
    2. k-core分解
    3. 最密子图算法（精确和近似）
    4. k-clique分解（极大团检测）
    5. 结果分析（无可视化）
    """
    
    def __init__(self, input_file=None):
        """
        初始化图对象
        
        Args:
            input_file: 输入文件路径
        """
        self.G = nx.Graph()
        self.original_nodes = {}  # 原始节点ID映射
        self.node_mapping = {}    # 节点映射字典
        self.reverse_mapping = {} # 反向映射字典
        
        if input_file:
            self.load_from_file(input_file)
    
    def load_from_file(self, filepath: str):
        """
        从文件读取图数据
        
        Args:
            filepath: 输入文件路径
        """
        print(f"正在读取文件: {filepath}")
        
        edges = []
        nodes_set = set()
        
        # 根据文件格式读取数据
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # 检查文件格式
        first_line = lines[0].strip()
        if first_line.startswith('#'):
            # Amazon格式：以#开头的注释行
            start_idx = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith('#'):
                    start_idx = i
                    break
        elif len(first_line.split()) == 2 and all(x.isdigit() for x in first_line.split()):
            # CondMat和Gowalla格式：第一行是节点数和边数
            n, m = map(int, first_line.split())
            start_idx = 1
            print(f"图信息: {n} 个节点, {m} 条边")
        else:
            start_idx = 0
        
        # 读取边信息
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    if u != v:  # 去除自环
                        edges.append((u, v))
                        nodes_set.add(u)
                        nodes_set.add(v)
                except ValueError:
                    continue
        
        print(f"读取到 {len(edges)} 条边, {len(nodes_set)} 个节点")
        
        # 创建节点映射（从原始ID到连续ID）
        sorted_nodes = sorted(nodes_set)
        self.node_mapping = {node: i for i, node in enumerate(sorted_nodes)}
        self.reverse_mapping = {i: node for i, node in enumerate(sorted_nodes)}
        
        # 构建图并去重边
        edge_set = set()
        for u, v in edges:
            mapped_u = self.node_mapping[u]
            mapped_v = self.node_mapping[v]
            # 确保无向边只添加一次
            edge = tuple(sorted([mapped_u, mapped_v]))
            edge_set.add(edge)
        
        # 添加边到图中
        self.G.add_edges_from(edge_set)
        
        print(f"构建完成: {self.G.number_of_nodes()} 个节点, {self.G.number_of_edges()} 条边")
        print(f"图密度: {self.density():.6f}")
        print(f"平均度: {self.average_degree():.2f}")
    
    def save(self, output_path: str):
        """
        保存图到文件
        
        Args:
            output_path: 输出文件路径
        """
        with open(output_path, 'w') as f:
            # 写入图的基本信息
            f.write(f"{self.G.number_of_nodes()} {self.G.number_of_edges()}\n")
            
            # 写入边信息（使用原始节点ID）
            for u, v in self.G.edges():
                orig_u = self.reverse_mapping[u]
                orig_v = self.reverse_mapping[v]
                f.write(f"{orig_u} {orig_v}\n")
        
        print(f"图已保存到: {output_path}")
    
    def density(self) -> float:
        """计算图的密度"""
        n = self.G.number_of_nodes()
        m = self.G.number_of_edges()
        if n <= 1:
            return 0.0
        return 2.0 * m / (n * (n - 1))
    
    def average_degree(self) -> float:
        """计算平均度"""
        if self.G.number_of_nodes() == 0:
            return 0.0
        return 2.0 * self.G.number_of_edges() / self.G.number_of_nodes()
    
    def k_cores(self, output_file: str):
        """
        k-core分解算法
        
        Args:
            output_file: 输出文件路径
        """
        print("开始k-core分解...")
        start_time = time.time()
        
        # 使用NetworkX的core_number函数
        core_numbers = nx.core_number(self.G)
        
        elapsed_time = time.time() - start_time
        
        # 写入结果
        with open(output_file, 'w') as f:
            f.write(f"{elapsed_time:.3f}s\n")
            
            # 按原始节点ID排序输出
            for mapped_node in sorted(core_numbers.keys()):
                orig_node = self.reverse_mapping[mapped_node]
                coreness = core_numbers[mapped_node]
                f.write(f"{orig_node} {coreness}\n")
        
        print(f"k-core分解完成，用时: {elapsed_time:.3f}s")
        print(f"结果保存到: {output_file}")
        
        # 统计信息
        max_core = max(core_numbers.values()) if core_numbers else 0
        print(f"最大core number: {max_core}")
        
        return core_numbers
    
    def densest_subgraph_exact(self, output_file: str):
        """
        精确最密子图算法（使用NetworkX实现）
        
        Args:
            output_file: 输出文件路径
        """
        print("开始精确最密子图计算...")
        start_time = time.time()
        
        try:
            # 使用NetworkX的densest_subgraph函数
            density, subgraph_nodes = nx.algorithms.approximation.densest_subgraph(self.G)
            
            elapsed_time = time.time() - start_time
            
            # 转换为原始节点ID
            orig_subgraph_nodes = [self.reverse_mapping[node] for node in subgraph_nodes]
            orig_subgraph_nodes.sort()
            
            # 写入结果
            with open(output_file, 'w') as f:
                f.write(f"{elapsed_time:.3f}s\n")
                f.write(f"{density:.6f}\n")
                f.write(" ".join(map(str, orig_subgraph_nodes)) + "\n")
            
            print(f"精确最密子图完成，用时: {elapsed_time:.3f}s")
            print(f"密度: {density:.6f}")
            print(f"子图大小: {len(subgraph_nodes)} 个节点")
            print(f"结果保存到: {output_file}")
            
            return density, subgraph_nodes
            
        except Exception as e:
            print(f"精确算法出错，使用近似算法: {e}")
            return self.densest_subgraph_approx(output_file)
    
    def densest_subgraph_approx(self, output_file: str):
        """
        2-近似最密子图算法（贪心删除算法）
        
        Args:
            output_file: 输出文件路径
        """
        print("开始2-近似最密子图计算...")
        start_time = time.time()
        
        # 复制图进行操作
        H = self.G.copy()
        best_density = 0.0
        best_subgraph = set(H.nodes())
        
        while H.number_of_edges() > 0:
            # 计算当前密度
            current_density = 2.0 * H.number_of_edges() / H.number_of_nodes() if H.number_of_nodes() > 0 else 0
            
            if current_density > best_density:
                best_density = current_density
                best_subgraph = set(H.nodes())
            
            # 找到度数最小的节点
            min_degree = float('inf')
            min_node = None
            for node in H.nodes():
                degree = H.degree(node)
                if degree < min_degree:
                    min_degree = degree
                    min_node = node
            
            if min_node is not None:
                H.remove_node(min_node)
            else:
                break
        
        elapsed_time = time.time() - start_time
        
        # 转换为原始节点ID
        orig_subgraph_nodes = [self.reverse_mapping[node] for node in best_subgraph]
        orig_subgraph_nodes.sort()
        
        # 写入结果
        with open(output_file, 'w') as f:
            f.write(f"{elapsed_time:.3f}s\n")
            f.write(f"{best_density:.6f}\n")
            f.write(" ".join(map(str, orig_subgraph_nodes)) + "\n")
        
        print(f"2-近似最密子图完成，用时: {elapsed_time:.3f}s")
        print(f"密度: {best_density:.6f}")
        print(f"子图大小: {len(best_subgraph)} 个节点")
        print(f"结果保存到: {output_file}")
        
        return best_density, best_subgraph
    
    def maximal_cliques(self, output_file: str):
        """
        使用Bron-Kerbosch算法寻找所有极大团
        
        Args:
            output_file: 输出文件路径
        """
        print("开始极大团检测...")
        start_time = time.time()
        
        # 使用NetworkX的find_cliques函数
        cliques = list(nx.find_cliques(self.G))
        
        elapsed_time = time.time() - start_time
        
        # 写入结果
        with open(output_file, 'w') as f:
            f.write(f"{elapsed_time:.3f}s\n")
            
            for clique in cliques:
                # 转换为原始节点ID并排序
                orig_clique = [self.reverse_mapping[node] for node in clique]
                orig_clique.sort()
                f.write(" ".join(map(str, orig_clique)) + "\n")
        
        print(f"极大团检测完成，用时: {elapsed_time:.3f}s")
        print(f"找到 {len(cliques)} 个极大团")
        
        # 统计信息
        if cliques:
            max_clique_size = max(len(clique) for clique in cliques)
            avg_clique_size = sum(len(clique) for clique in cliques) / len(cliques)
            print(f"最大团大小: {max_clique_size}")
            print(f"平均团大小: {avg_clique_size:.2f}")
        
        print(f"结果保存到: {output_file}")
        
        return cliques
    
    def k_clique_densest(self, k: int, output_file: str):
        """
        k-clique最密子图算法
        
        Args:
            k: clique大小
            output_file: 输出文件路径
        """
        print(f"开始{k}-clique最密子图计算...")
        start_time = time.time()
        
        # 找到所有大小至少为k的clique
        all_cliques = list(nx.find_cliques(self.G))
        k_cliques = [clique for clique in all_cliques if len(clique) >= k]
        
        if not k_cliques:
            print(f"没有找到大小至少为{k}的clique")
            with open(output_file, 'w') as f:
                f.write(f"{time.time() - start_time:.3f}s\n")
                f.write("0.0\n")
                f.write("\n")
            return
        
        # 计算每个k-clique的密度并找到最密的
        best_density = 0.0
        best_clique = []
        
        for clique in k_cliques:
            if len(clique) >= k:
                # 计算clique的密度
                subgraph = self.G.subgraph(clique)
                density = 2.0 * subgraph.number_of_edges() / subgraph.number_of_nodes()
                
                if density > best_density:
                    best_density = density
                    best_clique = clique
        
        elapsed_time = time.time() - start_time
        
        # 转换为原始节点ID
        orig_best_clique = [self.reverse_mapping[node] for node in best_clique]
        orig_best_clique.sort()
        
        # 写入结果
        with open(output_file, 'w') as f:
            f.write(f"{elapsed_time:.3f}s\n")
            f.write(f"{best_density:.6f} ")
            f.write(" ".join(map(str, orig_best_clique)) + "\n")
        
        print(f"{k}-clique最密子图完成，用时: {elapsed_time:.3f}s")
        print(f"密度: {best_density:.6f}")
        print(f"最佳{k}-clique大小: {len(best_clique)} 个节点")
        print(f"结果保存到: {output_file}")
        
        return best_density, best_clique
    
    def analyze_graph(self):
        """分析图的基础统计信息"""
        print("\n图结构分析:")
        print(f"节点数: {self.G.number_of_nodes()}")
        print(f"边数: {self.G.number_of_edges()}")
        print(f"图密度: {self.density():.6f}")
        print(f"平均度: {self.average_degree():.2f}")
        
        # 度数分布
        degrees = [self.G.degree(node) for node in self.G.nodes()]
        if degrees:
            print(f"最大度: {max(degrees)}")
            print(f"最小度: {min(degrees)}")
            print(f"度数中位数: {np.median(degrees):.2f}")
        
        # 连通性分析
        if nx.is_connected(self.G):
            print("图是连通的")
            print(f"直径: {nx.diameter(self.G)}")
            print(f"平均路径长度: {nx.average_shortest_path_length(self.G):.2f}")
        else:
            components = list(nx.connected_components(self.G))
            print(f"图不连通，有 {len(components)} 个连通分量")
            largest_component_size = max(len(comp) for comp in components)
            print(f"最大连通分量大小: {largest_component_size}")


def main():
    """主函数：演示图系统的使用"""
    print("=" * 60)
    print("算法课期末大作业 - 图结构挖掘系统 (无GUI版本)")
    print("=" * 60)
    
    # 处理三个数据集
    datasets = ["CondMat.txt", "Amazon.txt", "Gowalla.txt"]
    
    for dataset in datasets:
        print(f"\n处理数据集: {dataset}")
        print("-" * 40)
        
        try:
            # 创建图对象
            g = Graph(dataset)
            
            # 分析图结构
            g.analyze_graph()
            
            # 创建输出目录
            import os
            output_dir = f"output"
            os.makedirs(output_dir, exist_ok=True)
            
            dataset_name = dataset.replace('.txt', '')
            
            # 保存处理后的图
            g.save(f"{output_dir}/{dataset_name}_processed.txt")
            
            # 执行各种算法
            print(f"\n1. k-core分解")
            g.k_cores(f"{output_dir}/{dataset_name}_kcores.txt")
            
            print(f"\n2. 最密子图（精确算法）")
            g.densest_subgraph_exact(f"{output_dir}/{dataset_name}_densest_exact.txt")
            
            print(f"\n3. 最密子图（2-近似算法）")
            g.densest_subgraph_approx(f"{output_dir}/{dataset_name}_densest_approx.txt")
            
            # 对于较小的图，运行极大团算法
            if g.G.number_of_nodes() <= 10000:  # 限制节点数量避免计算过久
                print(f"\n4. 极大团检测")
                g.maximal_cliques(f"{output_dir}/{dataset_name}_maximal_cliques.txt")
                
                print(f"\n5. k-clique最密子图 (k=3)")
                g.k_clique_densest(3, f"{output_dir}/{dataset_name}_kclique_densest.txt")
            else:
                print(f"\n图太大({g.G.number_of_nodes()}节点)，跳过极大团算法")
            
        except Exception as e:
            print(f"处理 {dataset} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{dataset} 处理完成")


if __name__ == "__main__":
    main() 