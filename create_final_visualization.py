#!/usr/bin/env python3
"""
Create Final Visualization - Clear nodes with proper edge visibility
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graph_system import Graph
import os
import time

# Set matplotlib parameters
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

def create_clear_visualization(G, title, save_path, node_colors="lightblue", 
                              algorithm_type="original", special_nodes=None):
    """
    Create clear visualization with proper node and edge visibility
    
    Args:
        G: NetworkX graph object
        title: Graph title
        save_path: Save path
        node_colors: Node colors
        algorithm_type: Algorithm type
        special_nodes: Special nodes set
    """
    # Create large figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    print(f"📊 Generating visualization: {title}")
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Calculate layout - use spring layout for good distribution
    print("   Computing node layout...")
    pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42, 
                          weight=None, scale=1.0)
    
    # Optimized display parameters
    node_size = 200  # Larger nodes
    edge_alpha = 0.4  # More visible edges (increased from 0.15)
    edge_width = 1.2  # Slightly thicker edges
    
    # Set colors based on algorithm type
    if algorithm_type == "k_core" and isinstance(node_colors, list):
        # k-core decomposition: use color mapping
        print("   Applying k-core color mapping...")
        node_colors_final = plt.cm.viridis(np.array(node_colors))
        edge_color = '#666666'  # Darker gray edges
    elif algorithm_type == "densest" and special_nodes:
        # Densest subgraph: highlight special nodes
        print("   Highlighting densest subgraph nodes...")
        node_colors_final = []
        for node in G.nodes():
            if node in special_nodes:
                node_colors_final.append('#FF4444')  # Red
            else:
                node_colors_final.append('#87CEEB')  # Light blue
        edge_color = '#888888'  # Medium gray edges
    elif algorithm_type == "cliques" and special_nodes:
        # Maximal cliques: use different colors for cliques
        print("   Assigning colors to maximal cliques...")
        node_colors_final = ['#87CEEB'] * G.number_of_nodes()  # Default light blue
        node_list = list(G.nodes())
        
        # Assign different colors to the largest cliques
        clique_colors = ['#FF4444', '#44FF44', '#4444FF', '#FFAA44', '#FF44AA', '#44FFAA']
        sorted_cliques = sorted(special_nodes, key=len, reverse=True)[:6]
        
        for i, clique in enumerate(sorted_cliques):
            color = clique_colors[i % len(clique_colors)]
            for node in clique:
                if node in node_list:
                    idx = node_list.index(node)
                    node_colors_final[idx] = color
        edge_color = '#777777'  # Medium-dark gray edges
    else:
        # Original graph: uniform colors
        node_colors_final = node_colors if isinstance(node_colors, str) else '#87CEEB'
        edge_color = '#666666'  # Darker gray edges
    
    # Draw edges first (behind nodes)
    print("   Drawing edges...")
    nx.draw_networkx_edges(G, pos, 
                          alpha=edge_alpha,
                          width=edge_width,
                          edge_color=edge_color,
                          ax=ax)
    
    # Draw nodes second (on top of edges)
    print("   Drawing nodes...")
    node_collection = nx.draw_networkx_nodes(G, pos, 
                                           node_color=node_colors_final,
                                           node_size=node_size,
                                           alpha=0.9,
                                           linewidths=2.0,  # Thicker node borders
                                           edgecolors='black',
                                           ax=ax)
    
    # Add node labels (only for small graphs)
    if G.number_of_nodes() <= 80:
        print("   Adding node labels...")
        nx.draw_networkx_labels(G, pos, 
                              font_size=9, 
                              font_weight='bold',
                              font_color='black',
                              ax=ax)
    
    # Set title and style
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)
    ax.set_axis_off()
    
    # Add legend if needed
    if algorithm_type == "densest" and special_nodes:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF4444', label=f'Densest subgraph ({len(special_nodes)} nodes)'),
            Patch(facecolor='#87CEEB', label='Other nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    elif algorithm_type == "cliques" and special_nodes:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF4444', label='Clique 1 (largest)'),
            Patch(facecolor='#44FF44', label='Clique 2'),
            Patch(facecolor='#4444FF', label='Clique 3'),
            Patch(facecolor='#87CEEB', label='Other nodes')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    print(f"   Saving to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', 
               pad_inches=0.2)
    plt.close()
    
    print(f"✅ Visualization completed: {save_path}")

def analyze_and_visualize_graph(graph_file, output_dir="final_visualizations"):
    """
    Analyze graph and generate all visualizations
    """
    graph_name = os.path.basename(graph_file).replace('.txt', '')
    print(f"\n{'='*60}")
    print(f"📊 Analyzing graph: {graph_name}")
    print(f"{'='*60}")
    
    # Create output directory
    graph_output_dir = f"{output_dir}/{graph_name}"
    os.makedirs(graph_output_dir, exist_ok=True)
    
    # Load graph
    try:
        g = Graph(graph_file)
        print(f"📈 Graph loaded successfully:")
        print(f"   Nodes: {g.G.number_of_nodes()}")
        print(f"   Edges: {g.G.number_of_edges()}")
        print(f"   Density: {g.density():.6f}")
        print(f"   Average degree: {g.average_degree():.2f}")
        
        if nx.is_connected(g.G):
            print(f"   Connected: Yes")
            print(f"   Diameter: {nx.diameter(g.G)}")
        else:
            components = list(nx.connected_components(g.G))
            print(f"   Connected: No ({len(components)} components)")
    
    except Exception as e:
        print(f"❌ Graph loading failed: {e}")
        return
    
    # 1. Original graph visualization
    print(f"\n🎨 Creating original graph visualization...")
    create_clear_visualization(
        g.G,
        f"{graph_name} - Original Graph\n({g.G.number_of_nodes()} nodes, {g.G.number_of_edges()} edges)",
        f"{graph_output_dir}/{graph_name}_original.png",
        node_colors="#87CEEB",
        algorithm_type="original"
    )
    
    # 2. k-core decomposition
    print(f"\n🔍 Running k-core decomposition...")
    start_time = time.time()
    core_numbers = nx.core_number(g.G)
    k_core_time = time.time() - start_time
    
    max_core = max(core_numbers.values()) if core_numbers else 0
    print(f"   Execution time: {k_core_time:.3f}s")
    print(f"   Max core number: {max_core}")
    
    # Count core distribution
    core_dist = {}
    for node, core in core_numbers.items():
        core_dist[core] = core_dist.get(core, 0) + 1
    print(f"   Core distribution: {dict(sorted(core_dist.items()))}")
    
    # k-core visualization
    max_core_val = max(core_numbers.values()) if core_numbers else 1
    colors = []
    for node in g.G.nodes():
        core_num = core_numbers.get(node, 0)
        colors.append(core_num / max_core_val if max_core_val > 0 else 0)
    
    create_clear_visualization(
        g.G,
        f"{graph_name} - k-core Decomposition\n(Max core: {max_core}, Time: {k_core_time:.3f}s)",
        f"{graph_output_dir}/{graph_name}_kcore.png",
        node_colors=colors,
        algorithm_type="k_core"
    )
    
    # Save k-core results
    with open(f"{graph_output_dir}/{graph_name}_kcore_results.txt", 'w') as f:
        f.write(f"k-core Decomposition Results\n")
        f.write(f"Execution time: {k_core_time:.3f} seconds\n")
        f.write(f"Max core number: {max_core}\n\n")
        f.write("Node\tCore_Number\n")
        for mapped_node in sorted(core_numbers.keys()):
            orig_node = g.reverse_mapping[mapped_node]
            coreness = core_numbers[mapped_node]
            f.write(f"{orig_node}\t{coreness}\n")
    
    # 3. Densest subgraph algorithm
    print(f"\n🎯 Running densest subgraph algorithm...")
    start_time = time.time()
    try:
        density, subgraph_nodes = nx.algorithms.approximation.densest_subgraph(g.G)
        densest_time = time.time() - start_time
        
        print(f"   Execution time: {densest_time:.3f}s")
        print(f"   Subgraph density: {density:.6f}")
        print(f"   Subgraph size: {len(subgraph_nodes)} nodes")
        
        # Densest subgraph visualization
        create_clear_visualization(
            g.G,
            f"{graph_name} - Densest Subgraph\n(Density: {density:.3f}, Size: {len(subgraph_nodes)} nodes)",
            f"{graph_output_dir}/{graph_name}_densest.png",
            algorithm_type="densest",
            special_nodes=subgraph_nodes
        )
        
        # Save densest subgraph results
        orig_subgraph_nodes = [g.reverse_mapping[node] for node in subgraph_nodes]
        orig_subgraph_nodes.sort()
        
        with open(f"{graph_output_dir}/{graph_name}_densest_results.txt", 'w') as f:
            f.write(f"Densest Subgraph Results\n")
            f.write(f"Execution time: {densest_time:.3f} seconds\n")
            f.write(f"Density: {density:.6f}\n")
            f.write(f"Subgraph size: {len(subgraph_nodes)} nodes\n\n")
            f.write("Subgraph nodes:\n")
            for i, node in enumerate(orig_subgraph_nodes):
                if i % 10 == 0 and i > 0:
                    f.write("\n")
                f.write(f"{node} ")
    
    except Exception as e:
        print(f"   ❌ Densest subgraph algorithm failed: {e}")
    
    # 4. Maximal clique detection
    if g.G.number_of_nodes() <= 300:  # Only for medium-sized graphs
        print(f"\n🔺 Running maximal clique detection...")
        start_time = time.time()
        try:
            cliques = list(nx.find_cliques(g.G))
            cliques_time = time.time() - start_time
            
            if cliques:
                max_clique_size = max(len(clique) for clique in cliques)
                avg_clique_size = sum(len(clique) for clique in cliques) / len(cliques)
                
                print(f"   Execution time: {cliques_time:.3f}s")
                print(f"   Total cliques: {len(cliques)}")
                print(f"   Max clique size: {max_clique_size}")
                print(f"   Average clique size: {avg_clique_size:.2f}")
                
                # Maximal cliques visualization
                create_clear_visualization(
                    g.G,
                    f"{graph_name} - Maximal Cliques\n(Total: {len(cliques)}, Max size: {max_clique_size} nodes)",
                    f"{graph_output_dir}/{graph_name}_cliques.png",
                    algorithm_type="cliques",
                    special_nodes=cliques
                )
                
                # Save maximal cliques results
                with open(f"{graph_output_dir}/{graph_name}_cliques_results.txt", 'w') as f:
                    f.write(f"Maximal Cliques Detection Results\n")
                    f.write(f"Execution time: {cliques_time:.3f} seconds\n")
                    f.write(f"Total cliques: {len(cliques)}\n")
                    f.write(f"Max clique size: {max_clique_size}\n")
                    f.write(f"Average clique size: {avg_clique_size:.2f}\n\n")
                    f.write("Largest cliques:\n")
                    sorted_cliques = sorted(cliques, key=len, reverse=True)
                    for i, clique in enumerate(sorted_cliques[:10]):
                        orig_clique = [g.reverse_mapping[node] for node in clique]
                        orig_clique.sort()
                        f.write(f"Clique {i+1} (size {len(clique)}): {orig_clique}\n")
            
        except Exception as e:
            print(f"   ❌ Maximal cliques detection failed: {e}")
    else:
        print(f"\n🔺 Skipping maximal cliques (graph too large: {g.G.number_of_nodes()} nodes)")
    
    print(f"\n✅ {graph_name} analysis completed")

def main():
    """Main function"""
    print("🚀 Creating Final Visualization Demo")
    print("=" * 80)
    
    # Graph files to process
    demo_files = [
        "demo_graphs/demo_cliques_80.txt",     # Small clique structure graph
        "demo_graphs/demo_main_300_1000.txt", # Main demo graph
    ]
    
    output_dir = "final_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Processing {len(demo_files)} graph files:")
    for i, file in enumerate(demo_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    
    # Process each graph file
    for i, graph_file in enumerate(demo_files, 1):
        print(f"\n{'='*20} Processing {i}/{len(demo_files)} {'='*20}")
        try:
            analyze_and_visualize_graph(graph_file, output_dir)
        except Exception as e:
            print(f"❌ Processing {graph_file} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate report
    generate_report(output_dir)
    
    print(f"\n🎉 All visualizations created successfully!")
    print(f"📁 Results saved in: {output_dir}/")
    print(f"📊 Each graph includes:")
    print(f"   - Original graph visualization (clear nodes)")
    print(f"   - k-core decomposition visualization and results")
    print(f"   - Densest subgraph visualization and results")
    print(f"   - Maximal cliques visualization and results (small graphs)")

def generate_report(output_dir):
    """Generate visualization report"""
    report_file = f"{output_dir}/visualization_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Graph Mining System - Visualization Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report shows the visualization results of graph mining algorithms.\n")
        f.write("All visualizations are optimized to ensure nodes are clearly visible and not obscured by edges.\n\n")
        
        f.write("## Visualization Features\n\n")
        f.write("- **Clear node display**: Using larger node sizes with black borders\n")
        f.write("- **Visible edges**: Edge transparency set to 0.4 for good visibility\n")
        f.write("- **Color coding**: Different algorithms use different color schemes\n")
        f.write("- **High resolution output**: All images saved at 300 DPI\n\n")
        
        f.write("## Dataset Description\n\n")
        f.write("### demo_cliques_80\n")
        f.write("- **Nodes**: 80\n")
        f.write("- **Features**: Contains multiple obvious clique structures\n")
        f.write("- **Suitable algorithms**: All algorithms, especially maximal clique detection\n\n")
        
        f.write("### demo_main_300_1000\n")
        f.write("- **Nodes**: 300\n")
        f.write("- **Edges**: 1000\n")
        f.write("- **Features**: Contains community structures and hub nodes\n")
        f.write("- **Suitable algorithms**: k-core decomposition, densest subgraph\n\n")
        
        f.write("## Algorithm Results\n\n")
        
        # List results for each dataset
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                f.write(f"### {item}\n\n")
                
                files = [f for f in os.listdir(item_path) if f.endswith(('.png', '.txt'))]
                png_files = [f for f in files if f.endswith('.png')]
                txt_files = [f for f in files if f.endswith('.txt')]
                
                if png_files:
                    f.write("**Visualization files:**\n")
                    for file in sorted(png_files):
                        f.write(f"- {file}\n")
                    f.write("\n")
                
                if txt_files:
                    f.write("**Result files:**\n")
                    for file in sorted(txt_files):
                        f.write(f"- {file}\n")
                    f.write("\n")
    
    print(f"📋 Visualization report generated: {report_file}")

if __name__ == "__main__":
    main() 