import networkx as nx
import numpy as np
import math
import timeit

import matplotlib.pyplot as plt

print('Networkx version:', nx.__version__)

def get_graph_density(g):
    n_nodes = len(g.nodes())
    n_edges = len(g.edges())
    print('n_nodes:', n_nodes, ', n_edges:', n_edges)
    
    if nx.is_directed(g):
        density = n_edges / (n_nodes * (n_nodes - 1))
    else:
        density = 2 * n_edges / (n_nodes * (n_nodes - 1))
    
    return density


# Create an empty Directed Weighted Graph (DWG)
dwg = nx.DiGraph()

# Add nodes
nodes_list = [1, 2, 3, 4] # needs to be integers
dwg.add_nodes_from(nodes_list)

# Add labels to nodes (optional)
node_labels = ["A", "B", "C", "D"]

# Add weighted edges: format (node 1, node 2, cost)
edges_list = [(1, 2, 1), (1, 3, 1), (1, 4, 2), (3, 4, 1), (2, 3, 2)]
dwg.add_weighted_edges_from(edges_list)

# Selection of nodes, source and target
source = 4
target = 1

# Calculate the graph density
density = get_graph_density(dwg)
print('Graph density:', density)

# Using Dijkstra algorithm directly with the library
""" sp = nx.dijkstra_path(dwg, source, target)
print(sp) """

# From scratch
# Returns the node with a minimum own distance
def get_min_node(nodes, weights):
    min_node = -1
    min_weigth = math.inf
    
    for n in nodes:
        w = weights[n]
        if w < min_weigth:
            min_node = n
            min_weigth = w
    
    return min_node

# A detailed version of the Dijkstra algorithm for directed graphs with edges with positive weights 
def get_dijkstra_dist(graph, source, verbose=False):
    nodes = list(graph.nodes())
    edges = graph.edges()
    
    # Init distances
    dists = dict()
    for n in nodes:
        dists[n] = (0 if n == source else math.inf)
    paths = dict()
    for n in nodes:
        paths[n] = source
    
    # Greedy cycle
    v = source
    while len(nodes):        
        nodes.remove(v)
        if verbose:
            print('>> curr node:', v, ', len:', len(nodes))
        
        # Update weights
        for w in nodes:
            if (v, w) in edges:
                if dists[w] > dists[v] + edges[v, w]['weight']:
                    dists[w] = dists[v] + edges[v, w]['weight']
                    paths[w] = v
                    if verbose:
                        print('   v:', v, ', w:', w, ', weigth:', dists[w])
        
        # Get the node with a minimum own distance
        v = get_min_node(nodes, dists)
        if v == -1:
            break
        
    return { 'distances': dists, 'paths': paths }

# Show shortes path from source node to target node
def get_shortes_path(dwg, source, target, verbose=False):
    
    # Validation
    if not source in dwg.nodes() or not target in dwg.nodes():
        print('Both the source and the target must exist in the graph.')
        return {}
    
    start_time = timeit.default_timer()
    
    # Get the distance from 'source' to the other nodes
    sol = get_dijkstra_dist(dwg, source, verbose)
    paths = sol['paths']
    
    # Get shortest path from 'source' to 'target'
    ix = target
    path = [ix]
    while ix != source:
        ix = paths[ix]
        path.append(ix)
    path.reverse()
    
    weight = sol['distances'][target]
    
    # Elapsed time
    if verbose:
        elapsed = (timeit.default_timer() - start_time) * 1000
        print('>> elapsed time', elapsed, 'ms')
    
    return { 'path': path, 'weight': weight }

# Example of the shortes path calculation from '1' to '9'
sp_sol = get_shortes_path(dwg, source, target, True)
print(sp_sol)


# We then set the coordinates of each node (x, y)
dwg.nodes[1]['pos'] = (0, 0)
dwg.nodes[2]['pos'] = (1, 0)
dwg.nodes[3]['pos'] = (0, 1)
dwg.nodes[4]['pos'] = (1, 1)


# The positions of each node are stored in a dictionary
pos = nx.get_node_attributes(dwg, 'pos')

# Nodes labels
labels = {}
for n in dwg.nodes():
    try:
        if node_labels is not None:
            labels[n] = node_labels[n-1]
        else:
            labels[n] = n
    except NameError:
        labels[n] = n

# Edges labels
weights = {}
for u, v, w in dwg.edges(data=True):
    weights[(u, v)] = w['weight']

sp_edges = []
for i in range(len(sp_sol['path']) - 1):
    e = (sp_sol['path'][i], sp_sol['path'][i+1])
    sp_edges.append(e)

node_colors = []
#vtx_colors = graph_coloring_welsh_powell(upg, True)
for n in dwg.nodes():
    node_colors.append(int(255/(n+1/n)))

# Plot Directed Weighted Graph
plt.rcParams["figure.figsize"] = [10, 10]
nx.draw_networkx_nodes(dwg, pos, nodelist=dwg.nodes(), node_color=node_colors, node_size=500, alpha=1)
nx.draw_networkx_edges(dwg, pos, edgelist=dwg.edges(), width=1, alpha=0.8, edge_color='black')
nx.draw_networkx_edges(dwg, pos, edgelist=sp_edges,    width=3, alpha=0.9, edge_color='green')
nx.draw_networkx_labels(dwg, pos, labels, font_size=12, font_color='black')
nx.draw_networkx_edge_labels(dwg, pos, edge_labels=weights, font_color='black')
plt.title('Shortest Path of a DWG', fontsize=14)
plt.axis('off')
plt.show()
