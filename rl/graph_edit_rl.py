#!/usr/bin/env python
# coding: utf-8
# In[1]:
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#K.set_session(sess)
import itertools as it
import joblib as jl
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.constraints import NonNeg as nonneg
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
from tensorflow.keras.optimizers import Adam
import numpy as np
import numpy.random as npr
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import time
import copy
import rl.model_util as model_util



class ParameterSchedule(Callback):
    def __init__(self, 
                 constraint_coeff_mu,
                 constraint_coeff_mu_update_factors, 
                 budget,
                 constraint_epoch_start_schedule,
                 constraint_epsilon,
                 data_gen,
                 config):
        self.constraint_epoch_start_schedule = constraint_epoch_start_schedule
        self.constraint_epsilon = constraint_epsilon

        self.fair_loss_mu= K.variable(0)
        self.budget_loss_mu=K.variable(0)
        self.constraint_coeff_mu = constraint_coeff_mu
        
        
        self.flow_loss_switch = K.variable(0)
        self.fair_loss_switch = K.variable(0)
        self.budget_loss_switch = K.variable(0)
    
        self.constraint_coeff_mu_update_factor = constraint_coeff_mu_update_factors
        
        self.fair_loss_lambda = K.variable(0)
        self.edit_loss_lambda = K.variable(0)       
        
        self.budget = budget
        
        self.epoch_act = 0
        
        self.data_gen = data_gen
        self.config = config
        self.A_mask = None
        
        K.set_value(self.flow_loss_switch,1)
        K.set_value(self.fair_loss_switch,0)
        K.set_value(self.budget_loss_switch,0)
        self.mask = None
        
        self.temp = np.ones((1,1))
        
    def set_mask(self,A,mask):
        self.A = A
        self.mask = mask
        
    def set_temp(self,temp_var,temp_decay_factor):
        self.temp_var=temp_var
        self.temp_decay_factor = temp_decay_factor

    def set_group(self,g1,g2):
        self.g1=g1
        self.g2=g2

    def _schedule(self, epoch):
        if epoch>self.constraint_epoch_start_schedule[0]:
                K.set_value(self.fair_loss_mu,self.constraint_coeff_mu[0])
                K.set_value(self.fair_loss_switch,1)
         
        if epoch>self.constraint_epoch_start_schedule[1]:
                K.set_value(self.budget_loss_switch,1)
                K.set_value(self.budget_loss_mu,self.constraint_coeff_mu[1])
        
    
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        outputs = [[],[]]
        l1,l2=0,0
        if self.epoch_act>self.constraint_epoch_start_schedule[0] or self.epoch_act>self.constraint_epoch_start_schedule[1]:
            outputs = self.model.predict(next(self.data_gen)[0])
            
            outputs[0] = np.array(outputs[0])
            outputs[1] = np.array(outputs[1])
            vs = outputs[0]
            
            Vs = [np.mean(vs[i]) for i in range(len(vs))]
            #fairlosses = np.sum(np.abs(Vs - np.mean(Vs)))**self.fair_loss_p
            total=np.sum(Vs)
            r1 = len(self.g1) / (len(self.g1) + len(self.g2))
            Vgs = [total * r1, total * (1 - r1)]
            abs_diff = np.abs(np.array(Vs) - np.array(Vgs))
            l1=np.sum(abs_diff)
            Es = outputs[1]
            l2=np.mean(Es) - self.budget
            if l2<0:
                l2=0
        
            if self.epoch_act>self.constraint_epoch_start_schedule[0]:
            # Learning proxy lagrangian for fair loss
                
            #fairlosses = np.mean(fairlosses)
                fairlosses =  K.get_value(self.fair_loss_mu) * l1
            
                #K.set_value(self.fair_loss_lambda, (K.get_value(self.fair_loss_lambda) + fairlosses))
        
            if self.epoch_act>self.constraint_epoch_start_schedule[1]:

                edit_loss =  K.get_value(self.budget_loss_mu) * l2
               # K.set_value(self.edit_loss_lambda, (K.get_value(self.edit_loss_lambda) + edit_loss))

            #if np.sqrt(l1**2)>self.constraint_epsilon[0]:
                #K.set_value(self.fair_loss_mu, K.get_value(self.fair_loss_mu) * self.constraint_coeff_mu_update_factor)
            #if np.sqrt(l2**2)>self.constraint_epsilon[1]:
                #K.set_value(self.budget_loss_mu, K.get_value(self.budget_loss_mu) * self.constraint_coeff_mu_update_factor)
            
            if self.epoch_act>self.constraint_epoch_start_schedule[2]:
                    self.temp *= self.temp_decay_factor
                    K.set_value(self.temp_var, self.temp)
        self.epoch_act+=1
        self._schedule(self.epoch_act)

        print(f"Epoch {epoch}: loss = {logs['loss']}")


            
            
        
class FlowGraphEditRL():
    def __init__(self,config={}):
        ##graph
        #map_defaults(config, key, default)
        self.config = config
        g_s,mask, im,g1,g2,candidateedges = model_util.build_spa_graph(self.config)
        #g_s,mask, im,g1,g2,candidateedges = model_util.build_epinions_graph(self.config)
        #g_s,mask, im,g1,g2,candidateedges = model_util.build_erdos_renyi_graph(self.config)
        #g_s,mask, im,g1,g2,g3,candidateedges = model_util.build_facebook_graph(self.config)
        #self.g3=g3
        self.net_gs = g_s
        self.N = len(self.net_gs)
        self.immunized_ids = im
        self.mask = mask
        self.g1=g1
        self.g2=g2
        self.candidateedges=candidateedges

        #states
        states_fn = self.enforce_fn(self.map_defaults(config, "states fn","init_node_states"))
        states_params = self.map_defaults(config, "states params", {})
        self.states = states_fn(**states_params)
        self.num_states = len(self.states)

        #params
        self.num_groups = self.map_defaults(config, "num_groups",2)
        self.T = self.map_defaults(config, "T", 4)
        self.epochs = self.map_defaults(config, "epochs", 20)
        self.particles = self.map_defaults(config, "particles", 50)
        self.gamma = self.map_defaults(config, "gamma", 0.9)
        self.budget = self.map_defaults(config, "budget", 5)
        self.temp_decay_factor = self.map_defaults(config, "temp_decay_factor", 0.99)
        
        self.constraint_coeff_mu = self.map_defaults(config, "constraint_coeff_mu", [0.1,1])
        self.constraint_coeff_mu_update_factors = self.map_defaults(config, "constraint_coeff_mu_update_factors", 1.01)
        
        self.constraint_epoch_start_schedule = self.map_defaults(config, "constraint_epoch_start_schedule", [0,0,0])
        
        self.lr = self.map_defaults(config, "lr", 0.01)
        self.steps_per_epoch = self.map_defaults(config, "steps_per_epoch", 10)
        self.constraint_epsilon = self.map_defaults(config, "constraint_epsilon", [10,1])
        
        #particles
        self.exp_name  = self.map_defaults(config, "exp_name", 'exp')
        self.method = self.map_defaults(config, "method","exp")

        self.particle_params = self.map_defaults(config, "particle params", [{}, {}])#[{}, {}]
        self.particle_mc_params =  self.map_defaults(config,"particles MC", [{"particles":10000}, {"particles":10000}])
        #self.num_groups = len(self.particle_params)#len=2
        #self.particle_params_b = self.map_defaults(config, "particle params b", {})

        self.immunized_nodes = np.array(np.zeros(self.N))
        self.immunized_nodes[[im]] = 1
        self.new_immunized_nodes = np.array(np.ones(self.N))
        self.new_immunized_nodes[[im]] = 0

        self.param_schedule = ParameterSchedule(self.constraint_coeff_mu,
                                                self.constraint_coeff_mu_update_factors,
                                                self.budget,
                                                self.constraint_epoch_start_schedule,
                                                self.constraint_epsilon,
                                                #self.get_facebook_data_gen(),#facebook
                                                self.get_df_data_gen1(),
                                                self.config)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
        self.tbCallBack = TensorBoard(log_dir='../Graph/{}_{}_{}'.format(self.exp_name, self.method, timestamp))
        states_params = self.map_defaults(config, "state train params", {})
        self.dense_gs = None

        self.net_gs_trained = None
        self.train_history = None
        self.states_fn = states_fn
        self.states_params = states_params

        graph_edit_fn = self.enforce_fn(self.map_defaults(config, "graph edit fn","get_graph_edit_model"))
        graph_edit_params = self.map_defaults(config, "graph edit params", {})
        self.graph_edit_fn = graph_edit_fn
        self.graph_edit_params = graph_edit_params
        self.model_graph_edit_train, self.model_graph_edit_predict = graph_edit_fn(**graph_edit_params)



    def reset(self):
        self.model_graph_edit_train, self.model_graph_edit_predict = self.graph_edit_fn(**self.graph_edit_params)
        self.states_fn(**self.states_params)


            

    def train(self,config):
        if config['num_groups']==3:
            return self.train_facebook_graph()
        else:
            return self.train_fullydifferntiable1()

    
    def get_dfgroup_particles(self,group, particles=None):
        if particles is None:
            particles = self.particles
        if group==1:
            return  npr.choice([v for v in self.g1 if v not in self.immunized_ids],particles, replace=True)
        else:
            return  npr.choice([v for v in self.g2 if v not in self.immunized_ids],particles, replace=True)
        
    def get_df_data_gen1(self):
        Y = [np.array([0] * 1), np.array([0] * 1)]
        i = 0
        
        while True:
            locs = []
            locs.append(self.g1)
            locs.append(self.g2)
            state_gs = []
            for loc in locs:#2
                state_g = []
                v = np.zeros(self.N)
                v[[loc]]=1
                state_g.append(v)
                state_g = np.asarray(state_g)
                state_gs.append(state_g)

            i+=1            
            yield (state_gs,Y)
    

    def train_fullydifferntiable1(self):

        s0_all = np.zeros((self.num_states,self.num_states))
        for i in range(self.num_states):
            s0_all[i,i] = 1

        df_data_gen = self.get_df_data_gen1()
        Y = [np.array([0] * 1), np.array([0] * 1)]
        locs = []
        locs.append(self.g1)
        locs.append(self.g2)
        state_gs = []
        for loc in locs:#2
            state_g = []
            v = np.zeros(self.N)
            v[[loc]]=1
            state_g.append(v)
            state_g = np.asarray(state_g)
            state_gs.append(state_g)
        
        history = self.model_graph_edit_train.fit(state_gs,Y,
                                      # steps_per_epoch = self.steps_per_epoch,
                                       epochs=self.epochs, 
                                       callbacks=[self.param_schedule,self.tbCallBack,],
                                       verbose=1)
        
        locs = []
        locs.append(self.g1)
        locs.append(self.g2)
        state_gs = []
        for loc in locs:#2
            state_g = []
            v = np.zeros(self.N)
            v[[loc]]=1
            state_g.append(v)
            state_g = np.asarray(state_g)
            state_gs.append(state_g)
        
        for i in range(0,len(s0_all)):
            print('Predicted :{} of {}'.format(i,len(s0_all)))
            s0_sub=s0_all[i:min(i+1,len(s0_all))]
            
            E, Prs = self.model_graph_edit_predict.predict(state_gs)
            
            break
        print('E',E[0].shape)
        print('NUM',Prs)

        # ### Positions of edits (red folloed by black)
        #np.round(self.model_graph_edit.predict([S0_rs, S0_bs, Rs, temp])[1][:1], 0)
        self.net_gs_trained = E[0]
        self.train_history = history.history
        return E[0], history
    
    def get_facebook_data_gen(self):
        Y = [np.array([0] * 1), np.array([0] * 1)]
        i = 0
        
        while True:
            locs = []
            locs.append(self.g1)
            locs.append(self.g2)
            locs.append(self.g3)
            state_gs = []
            for loc in locs:#2
                state_g = []
                v = np.zeros(self.N)
                v[[loc]]=1
                state_g.append(v)
                state_g = np.asarray(state_g)
                state_gs.append(state_g)

            i+=1            
            yield (state_gs,Y)

    def train_facebook_graph(self):

        s0_all = np.zeros((self.num_states,self.num_states))
        for i in range(self.num_states):
            s0_all[i,i] = 1

        df_data_gen = self.get_facebook_data_gen()
        Y = [np.array([0] * 1), np.array([0] * 1)]
        locs = []
        locs.append(self.g1)
        locs.append(self.g2)
        locs.append(self.g3)
        state_gs = []
        for loc in locs:#2
            state_g = []
            v = np.zeros(self.N)
            v[[loc]]=1
            state_g.append(v)
            state_g = np.asarray(state_g)
            state_gs.append(state_g)
        
        history = self.model_graph_edit_train.fit(state_gs,Y,
                                      # steps_per_epoch = self.steps_per_epoch,
                                       epochs=self.epochs, 
                                       callbacks=[self.param_schedule,self.tbCallBack,],
                                       verbose=1)
        
        locs = []
        locs.append(self.g1)
        locs.append(self.g2)
        locs.append(self.g3)
        state_gs = []
        for loc in locs:#2
            state_g = []
            v = np.zeros(self.N)
            v[[loc]]=1
            state_g.append(v)
            state_g = np.asarray(state_g)
            state_gs.append(state_g)
        
        for i in range(0,len(s0_all)):
            s0_sub=s0_all[i:min(i+1,len(s0_all))]
            
            E, Prs = self.model_graph_edit_predict.predict(state_gs)
            
            break
        print('E',E[0].shape)
        print('NUM',Prs)

        # ### Positions of edits (red folloed by black)
        #np.round(self.model_graph_edit.predict([S0_rs, S0_bs, Rs, temp])[1][:1], 0)
        self.net_gs_trained = E[0]
        self.train_history = history.history
        return E[0], history

    def init_particle_locs_fb(self, d, pop_key=1, particles=None): # 1 = female, 2 = male
        if particles is None:
            particles = self.particles
        years = d["local_info"][:, 5]
        years_clean = [x for x,y in zip(*np.unique(years, return_counts=True)) if y > len(years)*.1 and x != 0]
        inds = [v for v in np.where(np.bitwise_and(d["local_info"][:, 1] == pop_key, d["local_info"][:, 5]==np.max(years_clean)))[0] if v not in self.immunized_ids]
        return npr.choice(inds, min(particles, len(inds)), replace=False)

    def init_node_states(self):
        return np.eye(self.N, dtype=int).tolist()

    def enforce_fn(self, default):
        if isinstance(default, str):
            default = getattr(self, default)
        print(default)
        return default

    def map_defaults(self, config, key, default):
        if key not in config:
            config[key] = default
            return default
        else:
            return config[key]

    def get_fd_particle_type_sub_graph(self,s, P_transpose, R,T):
            vs = []
            
            for i in range(T):
                r = kl.Lambda(lambda z: K.sum(z[0] * z[1], axis=-1, keepdims=True))([s, R])
                v = kl.Lambda(lambda z: (self.gamma ** i) * z)(r)
                vs.append(v)
                s = tf.matmul(s, P_transpose)
                s = kl.Lambda(lambda z: K.clip(z, 0, 3))(s)
                
            v = kl.Add()(vs)
            return v

        

    def get_graph_edit_model(self): #Done
            return self.get_graph_edit_fullydifferntiable_model()

    def get_graph_edit_fullydifferntiable_model(self): #Done
        
        sgs = []
        for i in range(self.num_groups):
            sgs.append(kl.Input((self.num_states,),name='sg{}'.format(i)))
        #print(kl.Input((self.num_states,)).shape) (bacth size,401)
        R1= np.expand_dims(np.array(self.immunized_nodes.tolist()), axis=0)
        self.temp_var = K.variable(np.ones((1,1)))
        self.param_schedule.set_temp(self.temp_var, self.temp_decay_factor)
        self.param_schedule.set_group(self.g1,self.g2)
        W = np.expand_dims(np.array(self.net_gs),axis=(0,-1))       
        Wd = np.expand_dims(np.zeros(np.array(self.net_gs).shape)* 0.5,axis=(0,-1))
        R_mat = np.expand_dims(np.diag(self.immunized_nodes), axis = (0,-1))
        A = np.expand_dims(np.array(self.mask), axis=(0,-1))
#
        
        
        W[W>0] = 1.
        print('# of editable edges:',np.sum(self.mask))
        print('Budget:',self.budget)
        print('number_group',self.num_groups)
        print('=========================================')
        R1 = kl.Input(tensor=K.constant(R1),shape=R1.shape, name="R")
        temp = kl.Input(tensor=self.temp_var, name="temp")
        print('temp',temp.shape)
        W = kl.Input(tensor=K.constant(W), name="W")
        print('W',W.shape)
        Wd = kl.Input(tensor=K.constant(Wd), name="Wd")
        print('Wd',Wd.shape)
        R_mat = kl.Input(tensor=K.constant(R_mat), name="R_mat")
        print('sgs',sgs[0].shape)
        E_features = kl.Concatenate(axis=-1)([R_mat,W])
        A = kl.Input(tensor=K.variable(A), name="A")
        self.param_schedule.set_mask(A, self.mask)

        dense_W = kl.Dense(self.num_states, name='dense_W')
        null_input = kl.Lambda(lambda z:0*z)(sgs[0])
        dense_W(null_input)
        W_d = dense_W.weights[0]
        yWd = kl.Lambda(lambda z:K.squeeze(z,axis=-1))(Wd)
        yWd = kl.Lambda(lambda z:z+W_d)(yWd)
        yWd = kl.Lambda(lambda z:K.expand_dims(z, axis=-1))(yWd)
        E_features = kl.Concatenate()([E_features,yWd])

        edge_inp = E_features#kl.Concatenate()([R,])
        for i in range(self.config["edge_num_layers"]):
            edge_inp = kl.Conv2D(self.config['hidden_dims'],
                                 self.config["edge_kernel_size"],
                                 strides=1,
                                 dilation_rate=i+1,
                                 activation='tanh',
                                 padding='same')(edge_inp)
        new_edges = kl.Conv2D(1,
                 1,
                 strides=1,
                 activation=None,
                 padding='same')(edge_inp)
        
        new_edges = kl.Lambda(lambda z:z[0]/z[1])([new_edges,temp])
        E = kl.Activation('sigmoid')(new_edges)
        #E = kl.Reshape((self.num_states,self.num_states))(E)
        E = kl.Lambda(lambda z:z[0]*z[1]*(1-z[2]))([E,A,W])
        
        
        W_effect = kl.Lambda(lambda z:K.squeeze(z[0] + z[1],axis=-1))([W,E])

        vgs = []
        for i in range(self.num_groups):
            vg = self.get_fd_particle_type_sub_graph(R1,W_effect[0],sgs[i],self.T)
            vgs.append(vg)

        vs = kl.Concatenate(name='value')(vgs)
        
        num_edits = kl.Lambda(lambda z:K.expand_dims(K.sum(z[0]*z[1],axis=(-1,-2,-3)),axis=-1))([A,E])
        num_edits = kl.Layer(name='edit')(num_edits)
        inps = sgs+[temp,R1,W,R_mat,A,Wd]


        model = km.Model(inputs=inps, outputs=[vs,num_edits])
        model.layers[-1].trainable_weights.extend([W_d])
        model.summary()
        
        model.compile(
            loss={
                'value':self.fair_flow_loss,
                'edit':self.budget_loss,
            }, 
            loss_weights={
                'value':1,
                'edit':1
            },
            optimizer=Adam(lr=self.lr),
            metrics = {
                'value':[model_util.get_flow_loss(), 
                         model_util.get_fair_loss_gourp2(self.config)],
                          #model_util.get_fair_lambda(self.param_schedule.fair_loss_lambda),
                          #model_util.get_edit_lambda(self.param_schedule.edit_loss_lambda),
                          #model_util.get_budget_loss_mu(self.param_schedule.budget_loss_mu),
                          #model_util.get_tempdecyfactor(self.temp_var)],
                'edit':[
                        model_util.get_num_edits(),
                        model_util.get_num_edits_exceeded(self.budget)
                        ]
            }
        )
        
        model_predict = km.Model(inputs=inps, outputs=[W_effect,num_edits])
        
        return model, model_predict

    def num_edits_exceeded_(self, y_true, y_pred):
        edits_exceeded = K.clip(K.mean(y_pred) - self.budget, 0 , 1e20)
        return edits_exceeded
    
    def budget_loss(self, y_true, y_pred):
        edit_loss = model_util.get_num_edits_exceeded(self.budget)(y_true, y_pred)
        #edit_loss = self.param_schedule.budget_loss_mu * (edit_loss**2) + 2 * self.param_schedule.edit_loss_lambda * edit_loss
        edit_loss = self.param_schedule.budget_loss_mu * (edit_loss) + 2 * self.param_schedule.edit_loss_lambda * edit_loss
        
        return edit_loss * self.param_schedule.budget_loss_switch

    def fair_loss_spectral_norm(self, y_true, y_pred):
        fair_loss = K.max(y_pred,axis=-1)-K.min(y_pred,axis=-1)
        print('^_^ fair_loss:',K.int_shape(fair_loss))
        return fair_loss
    
    def get_fairequity_loss(self):
        def fair_loss(y_true, y_pred):
            v = y_pred
            vgs = K.mean(v,axis=-2)
            vg_sum = K.sum(vgs, axis=-1)
            r1 = len(self.g1) / (len(self.g1) + len(self.g2))
            vg1 = vg_sum * r1
            vg2 = vg_sum * (1 - r1)
            vgs_modified = K.stack([K.abs(vgs[0] - vg1), K.abs(vgs[1] - vg2)], axis=-1)

            loss = K.sum(vgs_modified,axis=-1)
        
            return loss
        return fair_loss
    
    def get_group3_fair_loss(self):
        def fair_loss(y_true, y_pred):
            vg_sum=tf.reduce_sum(y_pred)
            r1 = len(self.g1) / (len(self.g1) + len(self.g2)+len(self.g3))
            r2 = len(self.g2) / (len(self.g1) + len(self.g2)+len(self.g3))
            r3 = len(self.g3) / (len(self.g1) + len(self.g2)+len(self.g3))
            vg1 = vg_sum * r1
            vg2 = vg_sum * r2
            vg3 = vg_sum * r3
            vgs_modified = K.stack([K.abs(y_pred[:, 0] - vg1), K.abs(y_pred[:, 1] - vg2),K.abs(y_pred[:, 2] - vg3)], axis=-1)

            loss = K.sum(vgs_modified,axis=-1)
        
            return loss
        return fair_loss
           
    def weighted_fair_loss(self, y_true, y_pred): 
        fair_loss = self.get_fairequity_loss()(y_true, y_pred) 
        #fair_loss = self.get_group3_fair_loss()(y_true, y_pred)
        fair_loss = self.param_schedule.fair_loss_mu * (fair_loss) + 2 * self.param_schedule.fair_loss_lambda * fair_loss
        
        return self.param_schedule.fair_loss_switch*fair_loss
        
    def fair_flow_loss(self, y_true, y_pred):

        return self.param_schedule.flow_loss_switch * model_util.get_flow_loss()(y_true, y_pred)+self.weighted_fair_loss(y_true, y_pred)
    

