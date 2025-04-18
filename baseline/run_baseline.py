import pickle
import random
import numpy as np
import baseline.shortestDistances as sd
import baseline.generateLPParams as gen
import networkx as nx
import joblib as jl
import networkx as nx
from collections import deque
from config_spa import *
#from config_email import *
#from config_epinions import *
def friendsOfFriendsNX(G):
    fof = {}
    for node in list(G.nodes()):
        fof[node] = set()
        for friend in list(G.neighbors(node)):
            for friendOfFriend in list(G.neighbors(friend)):
                if (not friendOfFriend in G.neighbors(node)) and (friendOfFriend != node):
                    fof[node].add(friendOfFriend)
    return fof

def IC(g, S,g1 ,p=0.2, mc=1000):
    S1=[]
    for i in range(len(S)):
        if S[i] in g:
            S1.append(S[i])
    S=S1
    spread = []
    sr1=[]
    sr2=[]
    for i in range(mc):
        new_active, A = S[:], S[:]
        r1=[]
        r2=[]
        while new_active:
            new_ones = []
            for node in new_active:
                np.random.seed(i)
                successors = list(g.successors(node)) 
                success = np.random.uniform(0, 1, len(successors)) < p
                new_ones += list(np.extract(success, successors))

            new_active = list(set(new_ones) - set(A))
            A += new_active
        for i in range(len(A)):
            if A[i] in g1:
                r1.append(A[i])
            else:
                r2.append(A[i])
        

        spread.append(len(A))
        sr1.append(len(r1))
        sr2.append(len(r2))
    return np.mean(sr1),np.mean(sr2)

def build_epinions_graph():
    edge_list_file = '/home/ubuntu/FIMA-master/Epinions_network/soc-Epinions1.txt'
    G = nx.read_edgelist(edge_list_file,create_using=nx.DiGraph())
    mapping = {str(node): int(node) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    random.seed(1) 
    selected_nodes = random.sample(G.nodes(), 1000)
    H = nx.induced_subgraph(G, selected_nodes)
    subgraph = G.subgraph(selected_nodes).copy()
    edges_in_subgraph = list(subgraph.edges())
    G1 = nx.DiGraph()
    G1.add_nodes_from(selected_nodes)
    G1.add_edges_from(edges_in_subgraph)
    new_nodes = range(0, len(list(G1.nodes())))
    mapping_relabel = dict(zip(list(G1.nodes()), new_nodes))
    G1 = nx.relabel_nodes(G1, mapping_relabel)
    nodes = list(G1.nodes())
    edges = set(G1.edges()) 
    random.shuffle(nodes)
    half = int(len(nodes) / 2.2)
    g1 = nodes[:half]
    g2 = nodes[half:]
    while len(edges) < 10000:
        if len(edges) > 8500:
            u, v = random.sample(g2, 2)
            if (u, v) not in edges and u != v:
                G1.add_edge(u, v)
                edges.add((u, v))
        else:
            u, v = random.sample(nodes, 2)
            if (u, v) not in edges:
                G1.add_edge(u, v)
                edges.add((u, v))

    return G1,g1,g2

def build_spa_graph():
    G = pickle.load(open('/home/ubuntu/FIMA-master/networks/graph_spa_500_0.pickle', 'rb'))
    G = nx.convert_node_labels_to_integers(G, label_attribute='pid') 
    nodeToGender = gen.getNoteToGenderNX(G)
    g1, g2 = gen.getGroups(nodeToGender, list(G.nodes()))
    return G,g1,g2


def remove_random_edges(G, p, g1, g2):
    np.random.seed(12)  # 
    edges = list(G.edges())  
    num_edges_to_remove = int(p * len(edges)) 
    cross_group_edges = [(u, v) for u, v in edges if (u in g1 and v in g2) or (u in g2 and v in g1)]
    same_group_edges = [(u, v) for u, v in edges if (u in g1 and v in g1) or (u in g2 and v in g2)]
    num_cross_group = int(0.5 * num_edges_to_remove)
    num_same_group = num_edges_to_remove - num_cross_group  
    edges_to_remove = random.sample(cross_group_edges, num_cross_group) + random.sample(same_group_edges, num_same_group)
    G.remove_edges_from(edges_to_remove)
    return G

def build_email_graph():
    edge_list_file = '/home/ubuntu/FIMA-master/email-Eu-core_network/email-Eu-core.txt'
    G = nx.read_edgelist(edge_list_file, create_using=nx.DiGraph())
    mapping = {str(node): int(node) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    node_labels = {}
    with open('/home/ubuntu/FIMA-master/email-Eu-core_network/email-Eu-core-department-labels.txt', 'r') as file:
        for line in file:
            node, label = map(int, line.split()) 
            node_labels[node] = label 
    g1=[]
    g2=[]
    for node, label in node_labels.items():
        if label % 2==1 :
            g1.append(node)
        else:
            g2.append(node)
    G=remove_random_edges(G,0.5,g1,g2)
    return G,g1,g2

def run_baseline_experiment(rledges):
    G,g1,g2=build_spa_graph()
    nodes = list(G.nodes())
    edges = []
    friendsOfFriends =friendsOfFriendsNX(G)
    for node in friendsOfFriends:
        for newNeighbor in friendsOfFriends[node]:
            edges.append((node, newNeighbor))
    print('number of candidate edges',len(edges))
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    sources=sorted_nodes[:config['seed_nodes']]
    print(sources)
    g1s, g2s =IC(G,sources,g1,0.1,1000)
    initial_value=g1s+g2s
    dis=(g1s/len(g1))/(g2s/len(g2))
    if dis>1:
        dis=1/dis
    print('initial_value',initial_value,'initial_fair',dis)
    G_rl = G.copy()
    G_rl.add_edges_from(rledges)
    g1rl,g2rl=IC(G_rl,sources,g1,0.1,1000)
    rl_value=g1rl+g2rl
    dis=(g1rl/len(g1))/(g2rl/len(g2))
    if dis>1:
        dis=1/dis
    print('rl_value',rl_value,'.rl_fair',dis,'rl_lift',(rl_value-initial_value)/initial_value)



















