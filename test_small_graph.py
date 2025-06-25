"""
测试图系统功能的小示例
"""
from graph_system import Graph
import os

def create_test_graph():
    """创建一个小的测试图"""
    # 创建一个简单的测试图文件
    test_data = """5 7
0 1
0 2
1 2
1 3
2 3
3 4
4 0
"""
    
    with open('test_graph.txt', 'w') as f:
        f.write(test_data)
    
    print("创建了测试图文件: test_graph.txt")
    print("图结构: 5个节点，7条边")

def test_all_algorithms():
    """测试所有算法功能"""
    print("=" * 50)
    print("测试图结构挖掘系统")
    print("=" * 50)
    
    # 创建测试图
    create_test_graph()
    
    # 加载图
    g = Graph('test_graph.txt')
    
    # 创建输出目录
    os.makedirs('test_output', exist_ok=True)
    
    # 测试所有算法
    print("\n1. 测试k-core分解...")
    g.k_cores('test_output/test_kcores.txt')
    
    print("\n2. 测试精确最密子图...")
    g.densest_subgraph_exact('test_output/test_densest_exact.txt')
    
    print("\n3. 测试2-近似最密子图...")
    g.densest_subgraph_approx('test_output/test_densest_approx.txt')
    
    print("\n4. 测试极大团检测...")
    g.maximal_cliques('test_output/test_maximal_cliques.txt')
    
    print("\n5. 测试k-clique最密子图...")
    g.k_clique_densest(3, 'test_output/test_kclique_densest.txt')
    
    print("\n6. 测试图可视化...")
    g.show("Test Graph")
    g.show_coreness()
    
    print("\n7. 保存处理后的图...")
    g.save('test_output/test_processed.txt')
    
    print("\n所有测试完成！")
    print("输出文件保存在 test_output/ 目录中")

if __name__ == "__main__":
    test_all_algorithms() 