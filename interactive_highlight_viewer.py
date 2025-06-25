#!/usr/bin/env python3
"""
Interactive Highlight Viewer - Click buttons to highlight algorithm results
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
from graph_system import Graph
import os

class InteractiveHighlightViewer:
    """Interactive viewer with algorithm result highlighting"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Graph Algorithm Highlight Viewer")
        self.root.geometry("1400x900")
        
        # Data storage
        self.graph = None
        self.pos = None
        self.algorithm_results = {}
        self.current_highlight = "original"
        
        # Create UI
        self.setup_ui()
        
        # Load default graph
        self.load_default_graph()
    
    def setup_ui(self):
        """Setup user interface"""
        # Top control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # File operations
        ttk.Button(control_frame, text="Load Graph", 
                  command=self.load_graph_file).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Load Test Dataset", 
                  command=self.load_demo_graph).pack(side=tk.LEFT, padx=5)
        
        # Algorithm buttons
        ttk.Label(control_frame, text="Highlight:").pack(side=tk.LEFT, padx=(20,5))
        
        ttk.Button(control_frame, text="Original Graph", 
                  command=lambda: self.highlight_algorithm("original")).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="k-core Results", 
                  command=lambda: self.highlight_algorithm("k_core")).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="Densest Subgraph", 
                  command=lambda: self.highlight_algorithm("densest")).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="Maximal Cliques", 
                  command=lambda: self.highlight_algorithm("cliques")).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(control_frame, text="k-clique Densest", 
                  command=lambda: self.highlight_algorithm("k_clique")).pack(side=tk.LEFT, padx=2)
        
        # Settings
        settings_frame = ttk.Frame(self.root)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(settings_frame, text="Node Size:").pack(side=tk.LEFT)
        self.node_size_var = tk.IntVar(value=100)
        ttk.Scale(settings_frame, from_=50, to=300, variable=self.node_size_var, 
                 orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(settings_frame, text="Edge Width:").pack(side=tk.LEFT, padx=(20,0))
        self.edge_width_var = tk.DoubleVar(value=1.5)
        ttk.Scale(settings_frame, from_=0.5, to=5.0, variable=self.edge_width_var, 
                 orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(settings_frame, text="Update Display", 
                  command=self.update_display).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(settings_frame, text="Save Current View", 
                  command=self.save_current_view).pack(side=tk.LEFT, padx=5)
        
        # Graph display area
        self.create_graph_display()
        
        # Info panel
        self.create_info_panel()
    
    def create_graph_display(self):
        """Create matplotlib graph display area"""
        graph_frame = ttk.Frame(self.root)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial display
        self.ax.text(0.5, 0.5, 'Load a graph to start', ha='center', va='center', 
                    transform=self.ax.transAxes, fontsize=16)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.canvas.draw()
    
    def create_info_panel(self):
        """Create information panel"""
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Current highlight info
        self.highlight_info = ttk.Label(info_frame, text="Current: Original Graph", 
                                       font=('Arial', 12, 'bold'))
        self.highlight_info.pack(side=tk.LEFT)
        
        # Graph stats
        self.stats_info = ttk.Label(info_frame, text="")
        self.stats_info.pack(side=tk.RIGHT)
    
    def load_graph_file(self):
        """Load graph from file"""
        filename = filedialog.askopenfilename(
            title="Select Graph File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.load_graph(filename)
    
    def load_demo_graph(self):
        """Load test graph"""
        demo_files = []
        if os.path.exists("test_datasets"):
            demo_files = [f"test_datasets/{f}" for f in os.listdir("test_datasets") 
                         if f.endswith('.txt')]
        
        if not demo_files:
            messagebox.showerror("Error", "No test datasets found")
            return
        
        # Simple selection dialog
        selection = tk.Toplevel(self.root)
        selection.title("Select Demo Graph")
        selection.geometry("400x300")
        
        tk.Label(selection, text="Select demo graph:", font=('Arial', 12)).pack(pady=10)
        
        listbox = tk.Listbox(selection, font=('Arial', 10))
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        for file in demo_files:
            listbox.insert(tk.END, os.path.basename(file))
        
        def on_select():
            sel = listbox.curselection()
            if sel:
                filename = demo_files[sel[0]]
                selection.destroy()
                self.load_graph(filename)
        
        tk.Button(selection, text="Load", command=on_select, 
                 font=('Arial', 10)).pack(pady=10)
    
    def load_default_graph(self):
        """Load default test graph if available"""
        default_graphs = [
            "test_datasets/multi_clique_80.txt",      # Small graph with cliques
            "test_datasets/community_300.txt",        # Medium graph with communities
            "test_datasets/scale_free_500.txt"        # Medium scale-free graph
        ]
        
        for graph_file in default_graphs:
            if os.path.exists(graph_file):
                self.load_graph(graph_file)
                break
    
    def load_graph(self, filename):
        """Load and analyze graph"""
        try:
            print(f"Loading graph: {filename}")
            
            # Load graph
            self.graph = Graph(filename)
            self.graph.filename = filename  # Store filename for result saving
            
            # Calculate layout once
            print("Computing layout...")
            if self.graph.G.number_of_nodes() > 500:
                # Sample large graphs
                sampled_nodes = list(np.random.choice(list(self.graph.G.nodes()), 
                                                    500, replace=False))
                self.display_graph = self.graph.G.subgraph(sampled_nodes)
            else:
                self.display_graph = self.graph.G
            
            self.pos = nx.spring_layout(self.display_graph, k=3.0, iterations=100, seed=42)
            
            # Run all algorithms
            self.run_all_algorithms()
            
            # Update display
            self.current_highlight = "original"
            self.update_display()
            
            # Update stats
            stats_text = f"Nodes: {self.graph.G.number_of_nodes()}, Edges: {self.graph.G.number_of_edges()}, Density: {self.graph.density():.4f}"
            self.stats_info.config(text=stats_text)
            
            print("Graph loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load graph: {e}")
            print(f"Error loading graph: {e}")
    
    def run_all_algorithms(self):
        """Run all algorithms and store results"""
        print("Running algorithms...")
        
        # Create output directories
        import os
        os.makedirs('algorithm_results', exist_ok=True)
        os.makedirs('static_images', exist_ok=True)
        
        self.algorithm_results = {}
        
        # Get graph name for file naming
        graph_name = os.path.basename(getattr(self.graph, 'filename', 'graph')).replace('.txt', '')
        
        try:
            import time
            
            # k-core decomposition
            print("  Running k-core decomposition...")
            start_time = time.time()
            core_numbers = nx.core_number(self.graph.G)
            elapsed_time = time.time() - start_time
            
            self.algorithm_results['k_core'] = {
                'core_numbers': core_numbers,
                'max_core': max(core_numbers.values()) if core_numbers else 0
            }
            
            # Save k-core results
            with open(f'algorithm_results/{graph_name}_kcore.txt', 'w') as f:
                f.write(f"k-core decomposition results\n")
                f.write(f"Execution time: {elapsed_time:.3f}s\n")
                f.write(f"Max core number: {max(core_numbers.values()) if core_numbers else 0}\n")
                f.write(f"Nodes and their core numbers:\n")
                for mapped_node in sorted(core_numbers.keys()):
                    orig_node = self.graph.reverse_mapping[mapped_node]
                    coreness = core_numbers[mapped_node]
                    f.write(f"{orig_node} {coreness}\n")
            
            # Densest subgraph
            print("  Running densest subgraph...")
            start_time = time.time()
            density, densest_nodes = nx.algorithms.approximation.densest_subgraph(self.graph.G)
            elapsed_time = time.time() - start_time
            
            self.algorithm_results['densest'] = {
                'density': density,
                'nodes': densest_nodes,
                'size': len(densest_nodes)
            }
            
            # Save densest subgraph results
            with open(f'algorithm_results/{graph_name}_densest.txt', 'w') as f:
                f.write(f"Densest subgraph results\n")
                f.write(f"Execution time: {elapsed_time:.3f}s\n")
                f.write(f"Density: {density:.6f}\n")
                f.write(f"Subgraph size: {len(densest_nodes)} nodes\n")
                f.write(f"Nodes in densest subgraph:\n")
                orig_nodes = [self.graph.reverse_mapping[node] for node in densest_nodes]
                orig_nodes.sort()
                f.write(" ".join(map(str, orig_nodes)) + "\n")
            
            # Maximal cliques (only for smaller graphs)
            if self.graph.G.number_of_nodes() <= 1000:
                print("  Running maximal cliques...")
                start_time = time.time()
                cliques = list(nx.find_cliques(self.graph.G))
                elapsed_time = time.time() - start_time
                
                self.algorithm_results['cliques'] = {
                    'cliques': cliques,
                    'count': len(cliques),
                    'max_size': max(len(c) for c in cliques) if cliques else 0
                }
                
                # Save cliques results
                with open(f'algorithm_results/{graph_name}_cliques.txt', 'w') as f:
                    f.write(f"Maximal cliques results\n")
                    f.write(f"Execution time: {elapsed_time:.3f}s\n")
                    f.write(f"Number of maximal cliques: {len(cliques)}\n")
                    f.write(f"Max clique size: {max(len(c) for c in cliques) if cliques else 0}\n")
                    f.write(f"All maximal cliques:\n")
                    for clique in cliques:
                        orig_clique = [self.graph.reverse_mapping[node] for node in clique]
                        orig_clique.sort()
                        f.write(" ".join(map(str, orig_clique)) + "\n")
            
            # k-clique densest (k=3)
            if self.graph.G.number_of_nodes() <= 1000:
                print("  Running k-clique densest...")
                start_time = time.time()
                all_cliques = list(nx.find_cliques(self.graph.G))
                k_cliques = [c for c in all_cliques if len(c) >= 3]
                
                if k_cliques:
                    best_density = 0.0
                    best_clique = []
                    
                    for clique in k_cliques:
                        subgraph = self.graph.G.subgraph(clique)
                        density = 2.0 * subgraph.number_of_edges() / subgraph.number_of_nodes()
                        if density > best_density:
                            best_density = density
                            best_clique = clique
                    
                    elapsed_time = time.time() - start_time
                    
                    self.algorithm_results['k_clique'] = {
                        'density': best_density,
                        'nodes': best_clique,
                        'size': len(best_clique)
                    }
                    
                    # Save k-clique results
                    with open(f'algorithm_results/{graph_name}_kclique.txt', 'w') as f:
                        f.write(f"k-clique densest subgraph results (k=3)\n")
                        f.write(f"Execution time: {elapsed_time:.3f}s\n")
                        f.write(f"Best density: {best_density:.6f}\n")
                        f.write(f"k-clique size: {len(best_clique)} nodes\n")
                        f.write(f"Nodes in k-clique densest subgraph:\n")
                        orig_clique = [self.graph.reverse_mapping[node] for node in best_clique]
                        orig_clique.sort()
                        f.write(" ".join(map(str, orig_clique)) + "\n")
            
            print("All algorithms completed!")
            print(f"Results saved to algorithm_results/ directory")
            
            # Auto-save static images for all algorithm results
            self.save_all_static_images()
            
        except Exception as e:
            print(f"Error running algorithms: {e}")
    
    def highlight_algorithm(self, algorithm):
        """Highlight specific algorithm results"""
        self.current_highlight = algorithm
        self.update_display()
        
        # Update info
        if algorithm == "original":
            self.highlight_info.config(text="Current: Original Graph")
        elif algorithm == "k_core":
            info = self.algorithm_results.get('k_core', {})
            max_core = info.get('max_core', 0)
            self.highlight_info.config(text=f"Current: k-core (Max core: {max_core})")
        elif algorithm == "densest":
            info = self.algorithm_results.get('densest', {})
            density = info.get('density', 0)
            size = info.get('size', 0)
            self.highlight_info.config(text=f"Current: Densest Subgraph (Density: {density:.3f}, Size: {size})")
        elif algorithm == "cliques":
            info = self.algorithm_results.get('cliques', {})
            count = info.get('count', 0)
            max_size = info.get('max_size', 0)
            self.highlight_info.config(text=f"Current: Maximal Cliques (Count: {count}, Max size: {max_size})")
        elif algorithm == "k_clique":
            info = self.algorithm_results.get('k_clique', {})
            density = info.get('density', 0)
            size = info.get('size', 0)
            self.highlight_info.config(text=f"Current: k-clique Densest (Density: {density:.3f}, Size: {size})")
    
    def update_display(self):
        """Update graph display with current highlighting"""
        if not self.graph or not self.pos:
            return
        
        self.ax.clear()
        
        # Get display parameters
        node_size = self.node_size_var.get()
        edge_width = self.edge_width_var.get()
        
        # Get node colors and edge highlighting based on current algorithm
        node_colors, edge_colors, highlighted_edges = self.get_highlight_colors()
        
        # Draw all edges first (background)
        nx.draw_networkx_edges(self.display_graph, self.pos, 
                              alpha=0.4, width=edge_width*0.7, 
                              edge_color='#666666', ax=self.ax)
        
        # Draw highlighted edges
        if highlighted_edges:
            nx.draw_networkx_edges(self.display_graph, self.pos, 
                                  edgelist=highlighted_edges,
                                  alpha=1.0, width=edge_width*2.0, 
                                  edge_color='#FF0000', ax=self.ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.display_graph, self.pos, 
                              node_color=node_colors,
                              node_size=node_size,
                              alpha=0.9,
                              linewidths=2.0,
                              edgecolors='black',
                              ax=self.ax)
        
        # Add labels for small graphs
        if self.display_graph.number_of_nodes() <= 50:
            labels = {node: self.graph.reverse_mapping[node] 
                     for node in self.display_graph.nodes()}
            nx.draw_networkx_labels(self.display_graph, self.pos, labels, 
                                   font_size=8, font_weight='bold', ax=self.ax)
        
        # Set title
        graph_name = getattr(self.graph, 'name', 'Graph')
        title = f"{graph_name} - {self.current_highlight.replace('_', ' ').title()} Highlighting"
        if hasattr(self, 'display_graph') and self.display_graph.number_of_nodes() < self.graph.G.number_of_nodes():
            title += f" (Showing {self.display_graph.number_of_nodes()}/{self.graph.G.number_of_nodes()} nodes)"
        
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        self.ax.set_axis_off()
        
        self.canvas.draw()
    
    def get_highlight_colors(self):
        """Get node colors and edge highlighting for current algorithm"""
        num_nodes = self.display_graph.number_of_nodes()
        node_colors = ['lightblue'] * num_nodes
        edge_colors = ['lightgray'] * self.display_graph.number_of_edges()
        highlighted_edges = []
        
        if self.current_highlight == "original":
            # Original graph - uniform colors
            pass
            
        elif self.current_highlight == "k_core" and 'k_core' in self.algorithm_results:
            # Color nodes by core number
            core_numbers = self.algorithm_results['k_core']['core_numbers']
            max_core = self.algorithm_results['k_core']['max_core']
            
            node_list = list(self.display_graph.nodes())
            colors = []
            for node in node_list:
                core_num = core_numbers.get(node, 0)
                intensity = core_num / max_core if max_core > 0 else 0
                colors.append(plt.cm.viridis(intensity))
            node_colors = colors
            
        elif self.current_highlight == "densest" and 'densest' in self.algorithm_results:
            # Highlight densest subgraph nodes and edges
            densest_nodes = self.algorithm_results['densest']['nodes']
            
            node_list = list(self.display_graph.nodes())
            colors = []
            for node in node_list:
                if node in densest_nodes:
                    colors.append('red')
                else:
                    colors.append('lightblue')
            node_colors = colors
            
            # Highlight edges within densest subgraph
            for edge in self.display_graph.edges():
                if edge[0] in densest_nodes and edge[1] in densest_nodes:
                    highlighted_edges.append(edge)
            
        elif self.current_highlight == "cliques" and 'cliques' in self.algorithm_results:
            # Color nodes by clique membership
            cliques = self.algorithm_results['cliques']['cliques']
            
            node_list = list(self.display_graph.nodes())
            colors = ['lightblue'] * len(node_list)
            
            # Color largest cliques with different colors
            sorted_cliques = sorted(cliques, key=len, reverse=True)[:5]
            clique_colors = ['red', 'green', 'orange', 'purple', 'brown']
            
            for i, clique in enumerate(sorted_cliques):
                color = clique_colors[i % len(clique_colors)]
                for node in clique:
                    if node in node_list:
                        idx = node_list.index(node)
                        colors[idx] = color
                        
                # Highlight edges within clique
                for j, node1 in enumerate(clique):
                    for node2 in clique[j+1:]:
                        if self.display_graph.has_edge(node1, node2):
                            highlighted_edges.append((node1, node2))
            
            node_colors = colors
            
        elif self.current_highlight == "k_clique" and 'k_clique' in self.algorithm_results:
            # Highlight k-clique densest result
            k_clique_nodes = self.algorithm_results['k_clique']['nodes']
            
            node_list = list(self.display_graph.nodes())
            colors = []
            for node in node_list:
                if node in k_clique_nodes:
                    colors.append('orange')
                else:
                    colors.append('lightblue')
            node_colors = colors
            
            # Highlight edges within k-clique
            for edge in self.display_graph.edges():
                if edge[0] in k_clique_nodes and edge[1] in k_clique_nodes:
                    highlighted_edges.append(edge)
        
        return node_colors, edge_colors, highlighted_edges
    
    def save_current_view(self):
        """Save current view to file"""
        if not self.graph:
            messagebox.showwarning("Warning", "No graph loaded")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"View saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {e}")
    
    def save_all_static_images(self):
        """Save static images for all algorithm results"""
        if not self.graph:
            return
        
        # Get graph name for file naming
        graph_name = os.path.basename(getattr(self.graph, 'filename', 'graph')).replace('.txt', '')
        
        algorithms = ["original", "k_core", "densest", "cliques", "k_clique"]
        algorithm_names = {
            "original": "Original Graph",
            "k_core": "k-core Results", 
            "densest": "Densest Subgraph",
            "cliques": "Maximal Cliques",
            "k_clique": "k-clique Densest"
        }
        
        print("Saving static images...")
        
        for algo in algorithms:
            # Skip algorithms that weren't run
            if algo != "original" and algo not in self.algorithm_results:
                continue
                
            # Set current highlight
            old_highlight = self.current_highlight
            self.current_highlight = algo
            
            # Update display
            self.update_display()
            
            # Save image
            filename = f"static_images/{graph_name}_{algo}.png"
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filename}")
            except Exception as e:
                print(f"  Error saving {filename}: {e}")
        
        # Restore original highlight
        self.current_highlight = old_highlight
        self.update_display()
        
        print("All static images saved to static_images/ directory")

def main():
    """Main function"""
    root = tk.Tk()
    app = InteractiveHighlightViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 