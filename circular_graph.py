# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 02:41:53 2024

@author: MaxGr
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Create a sample graph (replace with your own data)
G = nx.karate_club_graph()

# Circular layout
pos = nx.circular_layout(G)

# Draw the graph
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10)
plt.show()





# Create a sample graph (replace with your own data)
G = nx.karate_club_graph()

# Color mapping for nodes
node_colors = {node: (np.random.rand(), np.random.rand(), np.random.rand()) for node in G.nodes()}

# Color mapping for edges (optional)
edge_colors = [(0, 0, 0)] * len(G.edges())  # All black edges

# Circular layout
pos = nx.circular_layout(G)

# Set node colors
for node, color in node_colors.items():
    G.nodes[node]['color'] = color

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=800, node_color=[node_colors[node] for node in G.nodes()])

# Draw edges with colors
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=2.0, alpha=0.5)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

plt.axis('off')
plt.show()


