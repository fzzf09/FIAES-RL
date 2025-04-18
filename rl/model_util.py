from tensorflow.keras import backend as K
import numpy.random as npr
import numpy as np
import networkx as nx
import copy
import scipy.sparse as scs
import collections as c
import joblib as jl
import random
import time
import tensorflow as tf
import glob
from collections import deque
import heapq
import os
import scipy.io as sio
import igraph
import pickle
import pandas as pd
import baseline.shortestDistances as sd
import baseline.generateLPParams as gen
from scipy.sparse import csr_matrix

def get_fair_loss_gourp3(config):
    def fair_loss(y_true, y_pred):
        vg_sum=tf.reduce_sum(y_pred)
        vg1 = vg_sum * 0.3226
        vg2 = vg_sum * 0.5617
        vg3 = vg_sum * 0.1156
        vgs_modified = K.stack([K.abs(y_pred[:, 0] - vg1), K.abs(y_pred[:, 1] - vg2),K.abs(y_pred[:, 2] - vg3)], axis=-1)

        loss = K.sum(vgs_modified,axis=-1)
    
        return loss
    return fair_loss

def get_fair_loss_gourp2(config):
    def fair_loss(y_true, y_pred):
        vg_sum=tf.reduce_sum(y_pred)
        vg1 = vg_sum * 0.5
        vg2 = vg_sum * 0.5
        vgs_modified = K.stack([K.abs(y_pred[:, 0] - vg1), K.abs(y_pred[:, 1] - vg2)], axis=-1)
        loss = K.sum(vgs_modified,axis=-1)
        return loss
    return fair_loss

def diff_bw_groups(y_true, y_pred):
    N = K.int_shape(y_pred)[1]
    group_utilities = K.mean(y_pred,axis=0)
    mean_group_utilites = K.expand_dims(K.mean(group_utilities),axis=-1)
    diff_from_mean = K.abs(group_utilities-mean_group_utilites)
    diff_from_mean = K.sum(diff_from_mean)
    return diff_from_mean

def get_num_edits_exceeded(budget):
    def num_edits_exceeded(y_true, y_pred):
        edits_exceeded = K.clip((K.mean(y_pred) - budget), 0 , 1e20)
        return edits_exceeded
    return num_edits_exceeded
        


def get_tempdecyfactor(tempvar):
    def tempdecyfactor(y_true, y_pred):
        return  tempvar
    return tempdecyfactor

def get_fair_lambda(fair_loss_lambda):
    def get_fairlambda1(y_true, y_pred):
        return  fair_loss_lambda
    return  get_fairlambda1
def get_edit_lambda(edit_loss_lambda):
    def get_editlambda1(y_true, y_pred):
        return  edit_loss_lambda
    return  get_editlambda1

def get_budget_loss_mu(budget_loss_mu):
    def get_budget_loss_mu1(y_true, y_pred):
        return  budget_loss_mu
    return  get_budget_loss_mu1


def get_num_edits():
    def get_num_edits1(y_true, y_pred):
        #return   K.mean(y_pred)
        return y_pred
    return  get_num_edits1


def get_flow_loss():
    def flow_loss(y_true, y_pred):
        return -1 * K.pow((tf.reduce_sum(y_pred)),1)
    return flow_loss      

#wrapper for inner dictionary
def init_graph(N, adj_dim=2):
    l = [scs.dok_matrix((N, N)) for _ in range(adj_dim)] 
    ret = {"adjs" : l, "particles" : c.defaultdict(list), "immunized": set(), "N":N}
    return ret

#wrapper to translate networkx to dict (assumes undirected)
def networkx_wrapper(G, fn_edges=[lambda G, x,y: npr.exponential(2), lambda G, x,y: np.abs(npr.normal(0,1))], directed=True, self=False, aux = {}):

    ret = init_graph(np.max(G.nodes)+1, len(fn_edges))

    for v1, v2 in G.edges:
        for li, f_i in zip(ret["adjs"], fn_edges):
            li[v1, v2] = 1
    return ret

def networkx_wrapper1(G, fn_edges=[lambda G, x,y: npr.exponential(2), lambda G, x,y: np.abs(npr.normal(0,1))], directed=True, self=False, aux = {}):

    ret = init_graph(4039, len(fn_edges))

    for v1, v2 in G.edges:
        for li, f_i in zip(ret["adjs"], fn_edges):
            li[v1, v2] = 1
    return ret

def getNoteToGenderNX(G):
    nodeToGender = {}
    for node in list(G.nodes()):
        if G.nodes()[node]['gender'] == 'male':
            nodeToGender[node] = "1"
        else:
            nodeToGender[node] = "2"
    return nodeToGender
def friendsOfFriendsNX(G):
    fof = {}
    for node in list(G.nodes()):
        fof[node] = set()
        for friend in list(G.neighbors(node)):
            for friendOfFriend in list(G.neighbors(friend)):
                if (not friendOfFriend in G.neighbors(node)) and (friendOfFriend != node):
                    fof[node].add(friendOfFriend)
    return fof
def get_im_node(G,config):
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    im=sorted_nodes[:config['seed_nodes']]
    im= [int(x) for x in im]
    return im
def get_fof_graph(a,fof):
    a = np.array(a)
    a[:, -1] = 0
    g = igraph.Graph.Adjacency(a.astype(bool).tolist())
    g2 = igraph.Graph(directed=True)
    g2.add_vertices(g.vs.indices)
    es = [(i,fof[i][j]) for i in range(len(fof)) for j in range(len(fof[i]))]
    g2.add_edges(es)
    return g2.get_adjacency_sparse().toarray()
def getGroupsNames(nodeToGender, nodes):
    g1 = []
    g2 = []
    for node in nodes:
        if nodeToGender[node] == '1':
            g1.append(node)
        else:
            g2.append(node)
    return g1, g2
def getNoteToGenderNX(G):
    nodeToGender = {}
    for node in list(G.nodes()):
        if G.nodes()[node]['gender'] == 'male':
            nodeToGender[node] = "1"
        else:
            nodeToGender[node] = "2"
    return nodeToGender

def build_spa_graph(config):
    G = pickle.load(open('networks/'+config['used_graph'], 'rb'))
    G = nx.convert_node_labels_to_integers(G, label_attribute='pid')
    nodeToGender = getNoteToGenderNX(G)
    g1, g2 = getGroupsNames(nodeToGender, list(G.nodes()))
    fof = friendsOfFriendsNX(G)
    for node in fof:
        fof[node] = [x for x in fof[node] ]
    edges = []
    for node in fof:
        for newNeighbor in fof[node]:
            edges.append((node, newNeighbor))
    ret=networkx_wrapper(G,directed=True, self=False)
    im= get_im_node(G,config)
    shortestDistances = sd.multipleSourceShortestDistances(G, im)
    edgeNodeDistancesInit = gen.edgeToNodeDistancesUpdated(G, edges, shortestDistances)
    edges1 = [edges[edge] for edge in range(len(edges)) if len(edgeNodeDistancesInit[edge]) != 0]
    adj_matrix = nx.adjacency_matrix(G)
    prs = adj_matrix.toarray()
    num_nodes = len(G.nodes())
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for u, v in edges1:
        adj_matrix[int(u), int(v)] = 1
    mask = csr_matrix(adj_matrix).toarray()
    mask[(mask == 1) & (prs == 1)] = 0

    return prs,mask,im,g1,g2,edges1

def normalize_graph(g):
    g1 = g.sum(axis=1, keepdims=True)
    g1[np.isnan(g1)] = 1
    g1[g1 == 0] = 1
    return g / g1
def immunize(g, nodes):
    g = np.append(g, np.zeros([1, g.shape[1]]), axis = 0)
    g = np.append(g, np.zeros([g.shape[0], 1]), axis = 1)
    for i in list(nodes) + [g.shape[0]-1]:
        g[i, :] = 0
        g[i, -1] = 1
    return g

def immunize1(g, nodes):
    return g


def bfs_furthest_node(G, start):
    visited = {start}
    queue = deque([(start, 0)])  # (node, distance)
    furthest_node, max_distance = start, 0
    
    while queue:
        node, distance = queue.popleft()
        
        if distance > max_distance:
            furthest_node, max_distance = node, distance
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return furthest_node, max_distance

def find_longest_shortest_path_bfs(G):
    start_node = list(G.nodes())[0]
    
    furthest_node, _ = bfs_furthest_node(G, start_node)
    
    other_furthest_node, max_distance = bfs_furthest_node(G, furthest_node)
    
    return (furthest_node, other_furthest_node), max_distance

def add_longest_shortest_path_edges_infacebook(G, k):
    chosen_edge=[]
    for _ in range(k):
        node_pair,ds=find_longest_shortest_path_bfs(G)
        G.add_edge(*node_pair)
        chosen_edge.append(node_pair)
    return chosen_edge