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
# nodes_list = [1, 2, 3, 4] # needs to be integers
nodes_list = [i for i in range(1, 16)] # it will create nodes from 1 to 15
dwg.add_nodes_from(nodes_list)

# Add labels to nodes (optional)
node_labels = ["Clive's Office", 
               "Meeting Rom", 
               "Server", 
               "Nha's Office",
               "Hallway",
               "Printing room",
               "printing",
               "Arun's Office",
               "QLX Lab",
               "Hallway",
               "Exit",
               "Hallway",
               "Kitchen",
               "Ankit's office",
               "Ashton's office"
               ]

# Add weighted edges: format (node 1, node 2, cost)
edges_list = [(1, 4, 1), 
              (4, 5, 3),
              (5, 2, 1),
              (5, 6, 3),
              (6, 3, 1),
              (6, 7, 1),
              (4, 8, 1),
              (5, 10, 4),
              (10, 9, 3),
              (10, 11, 3),
              (10, 12, 2),
              (12, 13, 3),
              (12, 14, 1),
              (12, 15, 1)
              ]
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
dwg.nodes[1]['pos'] = (0, 8)
dwg.nodes[2]['pos'] = (1, 8)
dwg.nodes[3]['pos'] = (4, 8)
dwg.nodes[4]['pos'] = (0, 7)
dwg.nodes[5]['pos'] = (1, 7)
dwg.nodes[6]['pos'] = (4, 7)
dwg.nodes[7]['pos'] = (4, 6)
dwg.nodes[8]['pos'] = (0, 6)
dwg.nodes[9]['pos'] = (0, 3)
dwg.nodes[10]['pos'] = (1, 3)
dwg.nodes[11]['pos'] = (4, 3)
dwg.nodes[12]['pos'] = (1, 1)
dwg.nodes[13]['pos'] = (4, 1)
dwg.nodes[14]['pos'] = (0, 1)
dwg.nodes[15]['pos'] = (1, 0)

# The positions of each node are stored in a dictionary
pos = nx.get_node_attributes(dwg, 'pos')

# Nodes labels
labels = {}
for n in dwg.nodes():
    if node_labels:
        labels[n] = node_labels[n-1]
    else:
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
nx.draw_networkx_labels(dwg, pos, labels, font_size=12, font_color='w')
nx.draw_networkx_edge_labels(dwg, pos, edge_labels=weights, font_color='black')
plt.title('Shortest Path of a DWG', fontsize=14)
plt.axis('off')
plt.show()
