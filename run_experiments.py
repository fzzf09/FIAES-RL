import argparse
import random
import gc
import time
import collections as c
import sys
import glob
import networkx as nx
import os
import tensorflow as tf
import datetime
import pandas as pd
import numpy as np
from  scipy import stats
from collections import defaultdict
from tensorflow.keras import backend as K
import joblib as jl
import itertools as it
import matplotlib.pyplot as plt
import warnings
import scipy.io as sio
import tensorflow as tf
import rl.graph_edit_rl as grl
import baseline.run_baseline as rbl
#from config_epinions import *
#from config_erdos import *
#from config_email import *
from config_spa import *
#from config_facebook import *
if float(tf.version.VERSION[:3])>=2:
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior() 

warnings.filterwarnings("ignore")

def reset_keras():
    print(gc.collect()) 

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"

def set_device(gpu):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu);

def init_keras(device=0):
    set_device(device)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Run the RL method')

    args = parser.parse_args()
    run_type = args.graph
    experiment_type = args.exp

    init_keras(0)
    r = c.defaultdict(dict)
    config["states fn"] = "init_node_states"                                        
    config["states params"] = {}                                              
    config["graph edit fn"] = "get_graph_edit_model"                                 
    config["graph edit params"] = {}
    config["state train fn"] = "init_states_basic"                           
    config["state train params"] = {}
    config["no training"] = False
    config["evaluation iters"] = 1
    config["sample graphs iter"] = False
    config["param schedule"] = False
    config["evaluation baseline"] = True
      
    eval_log = c.defaultdict(list)
    model = grl.FlowGraphEditRL(config)  
    _, history = model.train(config)
    net_gs = model.net_gs
    net_gs_trained = model.net_gs_trained
    
    immunized = model.immunized_ids
    print(immunized)
    mask = model.mask                
    budget = model.budget
    rledges=[]
    edges_with_values = [
    (i, j, net_gs_trained[i][j])
    for i in range(len(net_gs_trained))
    for j in range(len(net_gs_trained))
    if net_gs[i][j] == 0
    ]

    top_k_edges = sorted(edges_with_values, key=lambda x: x[2], reverse=True)[:config['budget']]
    rledges = [(i, j) for i, j, _ in top_k_edges]
    rbl.run_baseline_experiment(rledges)
    model = None
    reset_keras()


    

