#!/usr/bin/env python3
"""
生成演示用的图数据 - 300个节点，1000条边
"""
import networkx as nx
import numpy as np
import random
import os

def generate_demo_graph():
    """生成300个节点、1000条边的演示图"""
    print("🎯 生成演示图数据 (300个节点, 1000条边)")
    
    # 设置随机种子以确保可重现
    random.seed(42)
    np.random.seed(42)
    
    # 创建图
    G = nx.Graph()
    
    # 添加节点 (1-300)
    nodes = list(range(1, 301))
    G.add_nodes_from(nodes)
    
    # 生成边 - 使用多种策略确保图有良好的结构
    edges_added = 0
    target_edges = 1000
    
    # 1. 首先创建一个连通的骨架 (生成树)
    print("📊 创建连通骨架...")
    # 使用随机生成树确保连通性
    for i in range(2, 301):
        # 随机选择一个已有的节点连接
        existing_node = random.randint(1, i-1)
        G.add_edge(existing_node, i)
        edges_added += 1
    
    print(f"   连通骨架: {edges_added} 条边")
    
    # 2. 添加一些社区结构
    print("📊 添加社区结构...")
    community_size = 50  # 每个社区50个节点
    num_communities = 6
    
    for comm_id in range(num_communities):
        start_node = comm_id * community_size + 1
        end_node = min((comm_id + 1) * community_size, 300)
        community_nodes = list(range(start_node, end_node + 1))
        
        # 在社区内添加额外的边
        community_edges_to_add = min(100, target_edges - edges_added)
        for _ in range(community_edges_to_add):
            if edges_added >= target_edges:
                break
            node1, node2 = random.sample(community_nodes, 2)
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2)
                edges_added += 1
    
    print(f"   社区结构: {edges_added} 条边")
    
    # 3. 添加一些hub节点 (高度节点)
    print("📊 添加hub节点...")
    hub_nodes = random.sample(nodes, 10)  # 选择10个hub节点
    
    for hub in hub_nodes:
        # 每个hub连接到15-25个其他节点
        connections_to_add = random.randint(10, 20)
        potential_targets = [n for n in nodes if n != hub and not G.has_edge(hub, n)]
        
        if len(potential_targets) > 0:
            targets = random.sample(potential_targets, 
                                  min(connections_to_add, len(potential_targets)))
            
            for target in targets:
                if edges_added >= target_edges:
                    break
                G.add_edge(hub, target)
                edges_added += 1
    
    print(f"   Hub节点: {edges_added} 条边")
    
    # 4. 随机添加剩余的边
    print("📊 添加随机边...")
    while edges_added < target_edges:
        node1, node2 = random.sample(nodes, 2)
        if not G.has_edge(node1, node2):
            G.add_edge(node1, node2)
            edges_added += 1
        
        # 防止无限循环
        if edges_added >= target_edges:
            break
    
    print(f"✅ 最终生成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    print(f"   密度: {nx.density(G):.6f}")
    print(f"   平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # 检查连通性
    if nx.is_connected(G):
        print("   ✅ 图是连通的")
        print(f"   直径: {nx.diameter(G)}")
    else:
        print("   ⚠️  图不连通")
        components = list(nx.connected_components(G))
        print(f"   连通分量数: {len(components)}")
    
    return G

def save_graph_to_file(G, filename):
    """将图保存到文件"""
    print(f"💾 保存图到文件: {filename}")
    
    with open(filename, 'w') as f:
        # 写入注释行
        f.write(f"# Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges\n")
        f.write(f"# Generated for algorithm demo\n")
        
        # 写入边
        for edge in G.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    
    print(f"✅ 图已保存到 {filename}")

def generate_small_clique_graph():
    """生成一个包含明显团结构的小图 (80个节点)"""
    print("🎯 生成小团结构图 (80个节点)")
    
    G = nx.Graph()
    
    # 创建几个大小不同的团
    clique_sizes = [12, 10, 8, 8, 6, 6, 5, 5]
    current_node = 1
    clique_nodes = []
    
    for clique_size in clique_sizes:
        # 创建一个完全图 (团)
        clique = list(range(current_node, current_node + clique_size))
        clique_nodes.append(clique)
        
        # 添加团内的所有边
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                G.add_edge(clique[i], clique[j])
        
        current_node += clique_size
    
    # 添加剩余的孤立节点
    remaining_nodes = 80 - current_node + 1
    for i in range(remaining_nodes):
        G.add_node(current_node + i)
    
    # 在团之间添加一些连接边
    for i in range(len(clique_nodes) - 1):
        # 每两个团之间连接1-2条边
        clique1 = clique_nodes[i]
        clique2 = clique_nodes[i + 1]
        
        # 随机选择节点连接
        for _ in range(random.randint(1, 2)):
            node1 = random.choice(clique1)
            node2 = random.choice(clique2)
            G.add_edge(node1, node2)
    
    # 添加一些随机边连接孤立节点
    isolated_nodes = list(range(current_node, 81))
    if isolated_nodes:
        for node in isolated_nodes:
            # 随机连接到某个团中的节点
            target_clique = random.choice(clique_nodes)
            target_node = random.choice(target_clique)
            G.add_edge(node, target_node)
    
    print(f"✅ 团结构图生成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    return G

def main():
    """主函数"""
    print("🚀 生成演示用图数据")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs("demo_graphs", exist_ok=True)
    
    # 1. 生成主要演示图 (300节点, 1000边)
    demo_graph = generate_demo_graph()
    save_graph_to_file(demo_graph, "demo_graphs/demo_main_300_1000.txt")
    
    print("\n" + "="*50)
    
    # 2. 生成小团结构图
    clique_graph = generate_small_clique_graph()
    save_graph_to_file(clique_graph, "demo_graphs/demo_cliques_80.txt")
    
    print(f"\n🎉 演示数据生成完成!")
    print(f"📁 文件保存在 demo_graphs/ 目录:")
    print(f"   - demo_main_300_1000.txt: 主要演示图 (300节点, ~1000边)")
    print(f"   - demo_cliques_80.txt: 团结构图 (80节点)")

if __name__ == "__main__":
    main() 