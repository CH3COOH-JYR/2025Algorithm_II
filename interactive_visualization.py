"""
图结构挖掘系统 - 交互式可视化界面
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
from graph_system import Graph
import os
import threading
import time

class GraphVisualizationApp:
    """图可视化交互应用"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("图结构挖掘系统 - 交互式可视化")
        self.root.geometry("1400x900")
        
        # 数据存储
        self.graph = None
        self.core_numbers = None
        self.cliques = None
        self.densest_subgraph = None
        self.current_layout = "spring"
        self.current_view = "original"
        
        # 创建界面
        self.create_widgets()
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        self.create_control_panel(main_frame)
        
        # 右侧图形显示区域
        self.create_graph_display(main_frame)
        
        # 底部状态栏
        self.create_status_bar()
    
    def create_control_panel(self, parent):
        """创建左侧控制面板"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="加载图文件", 
                  command=self.load_graph_file).pack(fill=tk.X, pady=2)
        
        ttk.Button(file_frame, text="选择测试数据", 
                  command=self.load_test_data).pack(fill=tk.X, pady=2)
        
        # 算法执行区域
        algo_frame = ttk.LabelFrame(control_frame, text="算法执行", padding=10)
        algo_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(algo_frame, text="k-core分解", 
                  command=self.run_k_cores).pack(fill=tk.X, pady=2)
        
        ttk.Button(algo_frame, text="最密子图", 
                  command=self.run_densest_subgraph).pack(fill=tk.X, pady=2)
        
        ttk.Button(algo_frame, text="极大团检测", 
                  command=self.run_cliques).pack(fill=tk.X, pady=2)
        
        # 可视化选项
        vis_frame = ttk.LabelFrame(control_frame, text="可视化选项", padding=10)
        vis_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 布局选择
        ttk.Label(vis_frame, text="图布局:").pack(anchor=tk.W)
        self.layout_var = tk.StringVar(value="spring")
        layout_combo = ttk.Combobox(vis_frame, textvariable=self.layout_var,
                                   values=["spring", "circular", "random", "kamada_kawai"])
        layout_combo.pack(fill=tk.X, pady=2)
        layout_combo.bind("<<ComboboxSelected>>", self.on_layout_change)
        
        # 视图选择
        ttk.Label(vis_frame, text="视图模式:").pack(anchor=tk.W, pady=(10, 0))
        
        self.view_var = tk.StringVar(value="original")
        views = [
            ("原始图", "original"),
            ("k-core着色", "k_core"),
            ("最密子图高亮", "densest"),
            ("极大团着色", "cliques")
        ]
        
        for text, value in views:
            ttk.Radiobutton(vis_frame, text=text, variable=self.view_var,
                           value=value, command=self.update_visualization).pack(anchor=tk.W)
        
        # 图信息显示
        info_frame = ttk.LabelFrame(control_frame, text="图信息", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=8, width=30)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 参数设置
        param_frame = ttk.LabelFrame(control_frame, text="参数设置", padding=10)
        param_frame.pack(fill=tk.X)
        
        # 节点大小
        ttk.Label(param_frame, text="节点大小:").pack(anchor=tk.W)
        self.node_size_var = tk.IntVar(value=50)
        node_size_scale = ttk.Scale(param_frame, from_=10, to=200, 
                                   variable=self.node_size_var, orient=tk.HORIZONTAL)
        node_size_scale.pack(fill=tk.X, pady=2)
        
        # 边宽度
        ttk.Label(param_frame, text="边宽度:").pack(anchor=tk.W)
        self.edge_width_var = tk.DoubleVar(value=1.0)
        edge_width_scale = ttk.Scale(param_frame, from_=0.1, to=3.0, 
                                    variable=self.edge_width_var, orient=tk.HORIZONTAL)
        edge_width_scale.pack(fill=tk.X, pady=2)
        
        # 更新按钮
        ttk.Button(param_frame, text="更新可视化", 
                  command=self.update_visualization).pack(fill=tk.X, pady=5)
    
    def create_graph_display(self, parent):
        """创建右侧图形显示区域"""
        graph_frame = ttk.Frame(parent)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图形
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # 创建canvas
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        toolbar.update()
    
    def create_status_bar(self):
        """创建底部状态栏"""
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_graph_file(self):
        """加载图文件"""
        filename = filedialog.askopenfilename(
            title="选择图文件",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if filename:
            self.load_graph(filename)
    
    def load_test_data(self):
        """选择测试数据"""
        test_dir = "test_datasets"
        if not os.path.exists(test_dir):
            messagebox.showerror("错误", "测试数据目录不存在，请先运行 generate_test_data.py")
            return
        
        files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]
        if not files:
            messagebox.showerror("错误", "测试数据目录为空")
            return
        
        # 创建选择对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("选择测试数据")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="选择要加载的测试数据:").pack(pady=10)
        
        listbox = tk.Listbox(dialog, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        for file in files:
            listbox.insert(tk.END, file)
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                filename = files[selection[0]]
                dialog.destroy()
                self.load_graph(os.path.join(test_dir, filename))
        
        ttk.Button(dialog, text="加载", command=on_select).pack(pady=10)
    
    def load_graph(self, filename):
        """加载图"""
        try:
            self.status_var.set(f"正在加载 {filename}...")
            self.root.update()
            
            # 使用Graph类加载
            self.graph = Graph(filename)
            
            # 重置算法结果
            self.core_numbers = None
            self.cliques = None
            self.densest_subgraph = None
            
            # 更新信息显示
            self.update_graph_info()
            
            # 更新可视化
            self.update_visualization()
            
            self.status_var.set(f"已加载图文件: {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图文件失败: {e}")
            self.status_var.set("加载失败")
    
    def update_graph_info(self):
        """更新图信息显示"""
        if not self.graph:
            return
        
        info = f"""图基本信息:
节点数: {self.graph.G.number_of_nodes()}
边数: {self.graph.G.number_of_edges()}
平均度: {self.graph.average_degree():.2f}
图密度: {self.graph.density():.6f}

连通性分析:"""
        
        if nx.is_connected(self.graph.G):
            info += f"""
图是连通的
直径: {nx.diameter(self.graph.G)}
平均聚类系数: {nx.average_clustering(self.graph.G):.4f}"""
        else:
            components = list(nx.connected_components(self.graph.G))
            largest_cc_size = max(len(comp) for comp in components)
            info += f"""
图不连通
连通分量数: {len(components)}
最大连通分量: {largest_cc_size}"""
        
        # 度数分布
        degrees = [self.graph.G.degree(node) for node in self.graph.G.nodes()]
        info += f"""

度数分布:
最大度: {max(degrees)}
最小度: {min(degrees)}
度数中位数: {np.median(degrees):.1f}"""
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
    
    def run_k_cores(self):
        """运行k-core分解"""
        if not self.graph:
            messagebox.showwarning("警告", "请先加载图文件")
            return
        
        def run_async():
            try:
                self.status_var.set("正在执行k-core分解...")
                self.root.update()
                
                self.core_numbers = nx.core_number(self.graph.G)
                
                # 更新信息
                max_core = max(self.core_numbers.values()) if self.core_numbers else 0
                info = f"\nk-core分解结果:\n最大core number: {max_core}"
                
                # 统计各core number的节点数
                core_dist = {}
                for node, core in self.core_numbers.items():
                    core_dist[core] = core_dist.get(core, 0) + 1
                
                info += "\ncore number分布:"
                for core in sorted(core_dist.keys()):
                    info += f"\ncore {core}: {core_dist[core]} 个节点"
                
                self.info_text.insert(tk.END, info)
                
                # 如果当前视图是k-core，更新可视化
                if self.view_var.get() == "k_core":
                    self.update_visualization()
                
                self.status_var.set("k-core分解完成")
                
            except Exception as e:
                messagebox.showerror("错误", f"k-core分解失败: {e}")
                self.status_var.set("k-core分解失败")
        
        # 在新线程中运行
        threading.Thread(target=run_async, daemon=True).start()
    
    def run_densest_subgraph(self):
        """运行最密子图算法"""
        if not self.graph:
            messagebox.showwarning("警告", "请先加载图文件")
            return
        
        def run_async():
            try:
                self.status_var.set("正在计算最密子图...")
                self.root.update()
                
                density, subgraph_nodes = nx.algorithms.approximation.densest_subgraph(self.graph.G)
                self.densest_subgraph = subgraph_nodes
                
                info = f"\n\n最密子图结果:\n密度: {density:.6f}\n子图大小: {len(subgraph_nodes)} 个节点"
                self.info_text.insert(tk.END, info)
                
                # 如果当前视图是最密子图，更新可视化
                if self.view_var.get() == "densest":
                    self.update_visualization()
                
                self.status_var.set("最密子图计算完成")
                
            except Exception as e:
                messagebox.showerror("错误", f"最密子图计算失败: {e}")
                self.status_var.set("最密子图计算失败")
        
        threading.Thread(target=run_async, daemon=True).start()
    
    def run_cliques(self):
        """运行极大团检测"""
        if not self.graph:
            messagebox.showwarning("警告", "请先加载图文件")
            return
        
        def run_async():
            try:
                self.status_var.set("正在检测极大团...")
                self.root.update()
                
                self.cliques = list(nx.find_cliques(self.graph.G))
                
                if self.cliques:
                    max_clique_size = max(len(clique) for clique in self.cliques)
                    avg_clique_size = sum(len(clique) for clique in self.cliques) / len(self.cliques)
                    
                    info = f"\n\n极大团检测结果:\n极大团数量: {len(self.cliques)}\n最大团大小: {max_clique_size}\n平均团大小: {avg_clique_size:.2f}"
                    
                    # 显示前几个较大的团
                    sorted_cliques = sorted(self.cliques, key=len, reverse=True)
                    info += "\n\n较大的极大团:"
                    for i, clique in enumerate(sorted_cliques[:5]):
                        orig_clique = [self.graph.reverse_mapping[node] for node in clique]
                        info += f"\n团{i+1} (大小{len(clique)}): {orig_clique[:10]}{'...' if len(clique) > 10 else ''}"
                else:
                    info = "\n\n极大团检测结果:\n未找到极大团"
                
                self.info_text.insert(tk.END, info)
                
                # 如果当前视图是极大团，更新可视化
                if self.view_var.get() == "cliques":
                    self.update_visualization()
                
                self.status_var.set("极大团检测完成")
                
            except Exception as e:
                messagebox.showerror("错误", f"极大团检测失败: {e}")
                self.status_var.set("极大团检测失败")
        
        threading.Thread(target=run_async, daemon=True).start()
    
    def on_layout_change(self, event=None):
        """布局改变时的回调"""
        self.current_layout = self.layout_var.get()
        self.update_visualization()
    
    def update_visualization(self):
        """更新可视化"""
        if not self.graph:
            return
        
        try:
            self.status_var.set("正在更新可视化...")
            self.root.update()
            
            # 清空画布
            self.ax.clear()
            
            # 获取子图进行可视化（对大图进行采样）
            G = self.graph.G
            if G.number_of_nodes() > 800:
                # 对大图进行采样
                sampled_nodes = list(np.random.choice(list(G.nodes()), 
                                                    min(800, G.number_of_nodes()), 
                                                    replace=False))
                G = G.subgraph(sampled_nodes)
                self.ax.set_title(f"图可视化 (采样显示 {len(sampled_nodes)} / {self.graph.G.number_of_nodes()} 个节点)")
            else:
                self.ax.set_title("图可视化")
            
            # 计算布局
            if self.current_layout == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif self.current_layout == "circular":
                pos = nx.circular_layout(G)
            elif self.current_layout == "random":
                pos = nx.random_layout(G)
            elif self.current_layout == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # 根据视图模式设置颜色
            node_colors = self.get_node_colors(G)
            
            # 绘制图
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                 node_size=self.node_size_var.get(), 
                                 alpha=0.8, ax=self.ax)
            
            nx.draw_networkx_edges(G, pos, alpha=0.4, 
                                 width=self.edge_width_var.get(), 
                                 ax=self.ax)
            
            # 如果节点不太多，显示标签
            if G.number_of_nodes() <= 50:
                # 使用原始节点ID作为标签
                labels = {node: self.graph.reverse_mapping[node] for node in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=self.ax)
            
            self.ax.set_axis_off()
            
            # 添加颜色条（如果需要）
            if self.view_var.get() == "k_core" and self.core_numbers:
                self.add_colorbar()
            
            # 刷新画布
            self.canvas.draw()
            
            self.status_var.set("可视化更新完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"可视化更新失败: {e}")
            self.status_var.set("可视化更新失败")
    
    def get_node_colors(self, G):
        """根据当前视图模式获取节点颜色"""
        view_mode = self.view_var.get()
        
        if view_mode == "original":
            return "lightblue"
        
        elif view_mode == "k_core" and self.core_numbers:
            # 根据core number着色
            max_core = max(self.core_numbers.values()) if self.core_numbers else 1
            colors = []
            for node in G.nodes():
                core_num = self.core_numbers.get(node, 0)
                colors.append(core_num / max_core if max_core > 0 else 0)
            return colors
        
        elif view_mode == "densest" and self.densest_subgraph:
            # 高亮最密子图
            colors = []
            for node in G.nodes():
                if node in self.densest_subgraph:
                    colors.append("red")
                else:
                    colors.append("lightblue")
            return colors
        
        elif view_mode == "cliques" and self.cliques:
            # 为不同的团分配不同颜色
            colors = ["lightblue"] * G.number_of_nodes()
            node_list = list(G.nodes())
            
            # 只显示前几个最大的团
            sorted_cliques = sorted(self.cliques, key=len, reverse=True)[:5]
            clique_colors = ["red", "green", "orange", "purple", "brown"]
            
            for i, clique in enumerate(sorted_cliques):
                color = clique_colors[i % len(clique_colors)]
                for node in clique:
                    if node in node_list:
                        idx = node_list.index(node)
                        colors[idx] = color
            
            return colors
        
        else:
            return "lightblue"
    
    def add_colorbar(self):
        """添加颜色条"""
        if self.view_var.get() == "k_core" and self.core_numbers:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            
            max_core = max(self.core_numbers.values())
            norm = mcolors.Normalize(vmin=0, vmax=max_core)
            sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
            sm.set_array([])
            
            cbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.6)
            cbar.set_label('Core Number')


def main():
    """主函数"""
    root = tk.Tk()
    app = GraphVisualizationApp(root)
    
    # 检查是否有测试数据
    if not os.path.exists("test_datasets"):
        messagebox.showinfo("提示", 
                          "未找到测试数据目录。\n"
                          "请先运行 'python generate_test_data.py' 生成测试数据，"
                          "或者手动加载图文件。")
    
    root.mainloop()


if __name__ == "__main__":
    main() 