
import rl.graph_edit_rl as grl
config = {}
config['num_groups']=2
config['dense_units']=128
config["epochs"] =2000
config["model_type"] = ['fullydifferntiable']
config["batch size state"] = 200                                                        #batch size for initial state  training
config["batch size train"] = 200                                                         #batch for graph edit training                                                             
config["gamma"] = 0.1                                                                   #fairness penalty weight
config["budget"] = 1800
config['seed_nodes']=6
config['constraint_coeff_mu']=[0.5,2.5]                                                                  #number of edits									# Coeff of squared constraint
config["constraint_coeff_mu_update_factors"] = 1.01						# Update factor of Coeff of squared constraint
config["constraint_epoch_start_schedule"] = [300,30,200]
config["constraint_epsilon"] = [10,1] #[0.03, 0.99]
config["hypothesis_difference_fraction"] = 0.05			# For unconstrainted optimization
config['used_graph']='graph_spa_500_0.pickle'
config["beta"] = 10
config["lr"] = 0.001
config["T"] = 15
config["T_max"] = 30
config["train iters"] = 300
config["steps_per_epoch"] = 5
config["evaluation replicates"] = 100
config["fair_loss_type"] = 'squared'
config["flow th"] = 0.0
config["hypothesis_difference_fraction"] = 0.05
config['temp_decay_factor']=0.999

config['exp_name']='exp_5e-9_0.9995_NoMeanaugLag-500'

config["group names"] = ["black", "red"]

config["fair_loss_p"] = 1.
config["fair_loss_normalize_variance"] = False

config["particle params"] =[{}, {}]
config["particles MC"] = [{"particles":1000}, {"particles":1000}]

config["hidden_dims"]=32
config["edge_kernel_size"]=5
config["edge_num_layers"]=3
config["batch_size"]=100
config['model_num_exp']=10

config['lagrangian_lamdba_growth']=0.005
config['lagrangian_lamdba2_growth']=0.1
config['evaluate_type']='IC'
