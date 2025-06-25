"""
生成测试数据集 - 创建具有不同特性的图
"""
import networkx as nx
import numpy as np
import random
from typing import List, Tuple
import os

def generate_scale_free_graph(n: int, m: int, seed: int = 42) -> nx.Graph:
    """
    生成无标度网络(scale-free network)
    
    Args:
        n: 节点数
        m: 每个新节点连接的边数
        seed: 随机种子
    
    Returns:
        NetworkX图对象
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 使用Barabási-Albert模型
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G

def generate_small_world_graph(n: int, k: int, p: float, seed: int = 42) -> nx.Graph:
    """
    生成小世界网络
    
    Args:
        n: 节点数
        k: 每个节点连接的最近邻居数
        p: 重连概率
        seed: 随机种子
    
    Returns:
        NetworkX图对象
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 使用Watts-Strogatz模型
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return G

def generate_community_graph(communities: List[int], p_in: float = 0.3, p_out: float = 0.05, seed: int = 42) -> nx.Graph:
    """
    生成带社区结构的图
    
    Args:
        communities: 每个社区的节点数列表
        p_in: 社区内连边概率
        p_out: 社区间连边概率
        seed: 随机种子
    
    Returns:
        NetworkX图对象
    """
    random.seed(seed)
    np.random.seed(seed)
    
    sizes = communities
    probs = [[p_out for _ in sizes] for _ in sizes]
    for i in range(len(sizes)):
        probs[i][i] = p_in
    
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return G

def generate_dense_regions_graph(n: int, num_dense_regions: int = 3, seed: int = 42) -> nx.Graph:
    """
    生成包含密集区域的图
    
    Args:
        n: 总节点数
        num_dense_regions: 密集区域数量
        seed: 随机种子
    
    Returns:
        NetworkX图对象
    """
    random.seed(seed)
    np.random.seed(seed)
    
    G = nx.Graph()
    
    # 分配节点到不同区域
    nodes_per_region = n // (num_dense_regions + 1)
    
    current_node = 0
    
    # 创建密集区域（团结构）
    for i in range(num_dense_regions):
        region_size = max(5, nodes_per_region // 2)
        region_nodes = list(range(current_node, current_node + region_size))
        
        # 在区域内创建高密度连接
        for u in region_nodes:
            for v in region_nodes:
                if u < v and random.random() < 0.7:  # 高连接概率
                    G.add_edge(u, v)
        
        current_node += region_size
    
    # 添加剩余节点，稀疏连接
    remaining_nodes = list(range(current_node, n))
    
    # 稀疏区域的连接
    for u in remaining_nodes:
        # 每个节点随机连接几个其他节点
        degree = random.randint(1, 4)
        candidates = [v for v in range(n) if v != u and not G.has_edge(u, v)]
        if candidates:
            targets = random.sample(candidates, min(degree, len(candidates)))
            for v in targets:
                G.add_edge(u, v)
    
    # 在密集区域之间添加一些桥连边
    dense_regions = []
    current = 0
    for i in range(num_dense_regions):
        region_size = max(5, nodes_per_region // 2)
        dense_regions.append(list(range(current, current + region_size)))
        current += region_size
    
    # 区域间连接
    for i in range(len(dense_regions)):
        for j in range(i + 1, len(dense_regions)):
            # 在不同区域间添加少量连边
            for _ in range(random.randint(1, 3)):
                u = random.choice(dense_regions[i])
                v = random.choice(dense_regions[j])
                G.add_edge(u, v)
    
    return G

def save_graph_to_file(G: nx.Graph, filename: str, original_format: bool = True):
    """
    保存图到文件
    
    Args:
        G: NetworkX图对象
        filename: 输出文件名
        original_format: 是否使用原始数据集格式
    """
    with open(filename, 'w') as f:
        if original_format:
            # 写入节点数和边数
            f.write(f"{G.number_of_nodes()} {G.number_of_edges()}\n")
        
        # 写入边信息
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    
    print(f"图已保存到: {filename}")
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    print(f"平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")

def analyze_graph(G: nx.Graph, name: str):
    """分析图的基本属性"""
    print(f"\n=== {name} 图分析 ===")
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    print(f"平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    
    # 度数分布
    degrees = [G.degree(node) for node in G.nodes()]
    print(f"最大度: {max(degrees)}")
    print(f"最小度: {min(degrees)}")
    
    # 连通性
    if nx.is_connected(G):
        print("图是连通的")
        print(f"直径: {nx.diameter(G)}")
        print(f"平均聚类系数: {nx.average_clustering(G):.4f}")
    else:
        components = list(nx.connected_components(G))
        print(f"图不连通，有 {len(components)} 个连通分量")
        largest_cc_size = max(len(comp) for comp in components)
        print(f"最大连通分量大小: {largest_cc_size}")
    
    # 密度
    density = nx.density(G)
    print(f"图密度: {density:.6f}")

def main():
    """生成各种测试数据集"""
    print("=" * 60)
    print("生成测试数据集")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs("test_datasets", exist_ok=True)
    
    # 1. 中等大小的无标度网络 (适合所有算法)
    print("\n1. 生成无标度网络...")
    sf_graph = generate_scale_free_graph(n=500, m=3, seed=42)
    analyze_graph(sf_graph, "无标度网络")
    save_graph_to_file(sf_graph, "test_datasets/scale_free_500.txt")
    
    # 2. 小世界网络
    print("\n2. 生成小世界网络...")
    sw_graph = generate_small_world_graph(n=400, k=6, p=0.3, seed=42)
    analyze_graph(sw_graph, "小世界网络")
    save_graph_to_file(sw_graph, "test_datasets/small_world_400.txt")
    
    # 3. 社区结构图
    print("\n3. 生成社区结构图...")
    community_graph = generate_community_graph(
        communities=[80, 90, 70, 60], 
        p_in=0.4, 
        p_out=0.02, 
        seed=42
    )
    analyze_graph(community_graph, "社区结构图")
    save_graph_to_file(community_graph, "test_datasets/community_300.txt")
    
    # 4. 包含密集区域的图
    print("\n4. 生成密集区域图...")
    dense_graph = generate_dense_regions_graph(n=350, num_dense_regions=4, seed=42)
    analyze_graph(dense_graph, "密集区域图")
    save_graph_to_file(dense_graph, "test_datasets/dense_regions_350.txt")
    
    # 5. 更大的图 (用于测试算法性能)
    print("\n5. 生成较大规模图...")
    large_graph = generate_scale_free_graph(n=1000, m=4, seed=42)
    analyze_graph(large_graph, "大规模无标度网络")
    save_graph_to_file(large_graph, "test_datasets/large_scale_free_1000.txt")
    
    # 6. 包含多个团的图
    print("\n6. 生成多团结构图...")
    clique_graph = nx.Graph()
    
    # 创建几个不同大小的团
    clique_sizes = [8, 6, 7, 5, 9]
    current_node = 0
    clique_nodes = []
    
    for size in clique_sizes:
        nodes = list(range(current_node, current_node + size))
        clique_nodes.append(nodes)
        
        # 创建完全图（团）
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                clique_graph.add_edge(nodes[i], nodes[j])
        
        current_node += size
    
    # 在团之间添加一些连边
    for i in range(len(clique_nodes)):
        for j in range(i + 1, len(clique_nodes)):
            # 在不同团之间添加1-2条边
            for _ in range(random.randint(1, 2)):
                u = random.choice(clique_nodes[i])
                v = random.choice(clique_nodes[j])
                clique_graph.add_edge(u, v)
    
    # 添加一些随机节点和边
    extra_nodes = 50
    for node in range(current_node, current_node + extra_nodes):
        # 每个额外节点连接到1-3个已有节点
        degree = random.randint(1, 3)
        targets = random.sample(list(clique_graph.nodes()), min(degree, len(clique_graph.nodes())))
        for target in targets:
            clique_graph.add_edge(node, target)
    
    analyze_graph(clique_graph, "多团结构图")
    save_graph_to_file(clique_graph, "test_datasets/multi_clique_80.txt")
    
    print(f"\n所有测试数据集已生成到 test_datasets/ 目录")
    print("\n推荐使用的数据集：")
    print("- scale_free_500.txt: 中等规模，适合所有算法")
    print("- community_300.txt: 社区结构明显，适合k-core分析")
    print("- dense_regions_350.txt: 包含密集区域，适合最密子图算法")
    print("- multi_clique_80.txt: 包含多个团，适合极大团算法")
    print("- large_scale_free_1000.txt: 大规模测试")

if __name__ == "__main__":
    main() 