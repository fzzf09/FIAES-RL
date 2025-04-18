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

def build_spa_graph():
    G = pickle.load(open('/home/ubuntu/FIMA-master/networks/graph_spa_500_0.pickle', 'rb'))
    G = nx.convert_node_labels_to_integers(G, label_attribute='pid') 
    nodeToGender = gen.getNoteToGenderNX(G)
    g1, g2 = gen.getGroups(nodeToGender, list(G.nodes()))
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



















