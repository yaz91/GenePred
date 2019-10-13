#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scipy.io as sio
import time
import copy
import string
import numpy as np
import pickle
import tensorflow as tf

from config import Config
from graph import Graph
from utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

config = Config("SaccharomycesCerevisiaeGene.txt","SaccharomycesCerevisiaeAnnonation.txt","SaccharomycesCerevisiaeSequence.txt")

config.struct = [None,400]
config.alpha = 500
config.dropout = [0.1,0.1]
config.loops = 3
config.loops_H = 10
config.loss_gain = 0.5
config.gain = 0.2
config.learning_rate = 0.001
config.g_lambda = -1
config.g_lambda0 = 500
# config.g_lambda1 = 10
config.g_regX = 5e-3
config.g_regH = 0
config.g_learning_rate = 0.0001
config.g_stddev = 1e-3
config.sample_ratio = 1
config.sample_method = 'node'

graph = Graph(config.data_graph,config.ng_sample_ratio)
graph.load_label_data(config.label_file_path)
graph.adj_matrix = np.where(graph.adj_matrix>=1,1,0)
graph.load_sequence_data(config.sequence_path)
origin_data = copy.deepcopy(graph)
graph = graph.subgraph(config.sample_method, config.sample_ratio)

config.struct[0] = origin_data.N
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config =  tf_config)


# In[3]:


###############################################
############ define variables ##################

layers = len(config.struct)
W = {}
b = {}
for i in range(layers - 1):
    name = "encoder" + str(i)
    W[name] = tf.Variable(tf.random_normal([config.struct[i], config.struct[i+1]]), name = name)
    b[name] = tf.Variable(tf.zeros([config.struct[i+1]]), name = name)

config.struct.reverse()
for i in range(layers - 1):
    name = "decoder" + str(i)
    W[name] = tf.Variable(tf.random_normal([config.struct[i], config.struct[i+1]]), name = name)
    b[name] = tf.Variable(tf.zeros([config.struct[i+1]]), name = name)
config.struct.reverse()

###############################################
############## define input ###################

adjacent_matriX = tf.placeholder("float", [None, None])
X_standard = tf.placeholder("float", [None, config.struct[0]])
H_ref = tf.placeholder("float", [None, config.struct[-1]])
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)

###############################################
########### define loss&structure #############

def encoder(X,dropout = np.zeros([layers-1,]),is_training = False):
    for i in range(layers - 1):
        name = "encoder" + str(i)
        X = tf.layers.dropout(tf.nn.sigmoid(tf.matmul(X, W[name]) + b[name]),rate = dropout[i],training = is_training)
    return X

def decoder(X,dropout = np.zeros([layers-1,]),is_training = False):
    for i in range(layers - 1):
        name = "decoder" + str(i)
        X =  tf.layers.dropout(tf.nn.sigmoid(tf.matmul(X, W[name]) + b[name]),rate = dropout[layers-2-i],training = is_training)
    return X

def get_1st_loss(H, adj_mini_batch,beta_l):
    D = tf.diag(tf.reduce_sum(adj_mini_batch+beta_l,1))
    L = D - (adj_mini_batch+beta_l) ## L is laplation-matriX
    return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(H),L),H))

def get_2nd_loss(X, newX, beta):
    B = X * (beta - 1) + 1
    return tf.reduce_sum(tf.pow((newX - X)* B, 2))

def get_label_loss(X,X_label):
    X_mean = tf.div(tf.matmul(X,tf.transpose(X_label)),tf.reduce_sum(X_label))
    return 0

def get_g_loss(Hr,Hd,XrOH,Xs,beta):
    B = Xs * (beta - 1) + 1
    return config.g_lambda0*tf.reduce_sum(tf.pow(Hr-Hd, 2))+config.g_lambda*tf.reduce_sum(tf.pow((Xs-XrOH)*B, 2))+config.g_regX*tf.contrib.layers.apply_regularization(l1_regularizer, [XrOH])+config.g_regH*tf.contrib.layers.apply_regularization(l1_regularizer, [Hr])

def get_g_loss_rand(Hr,Hd,Hp,XrOH,Xs):
    return config.g_lambda0*tf.reduce_sum(tf.pow(Hr-Hd, 2))+config.g_lambda*tf.reduce_sum(tf.pow(Xs-XrOH, 2))+config.g_regX*tf.contrib.layers.apply_regularization(l1_regularizer, [XrOH])+config.g_regH*tf.contrib.layers.apply_regularization(l1_regularizer, [Hr])+config.g_lambda1*tf.reduce_sum(tf.trace(tf.matmul(Hr-Hd,Hp-Hd,transpose_b=True)))

# feature extracter
H = encoder(X_standard,config.dropout,config.is_training)
X_reconstruct = decoder(H,config.dropout,config.is_training)
loss_2nd = get_2nd_loss(X_standard, X_reconstruct, config.beta)
loss_1st = get_1st_loss(H, adjacent_matriX,config.beta_l)
loss_xxx = tf.reduce_sum(tf.pow(X_reconstruct,2))
d_loss = config.gamma * loss_1st + config.alpha * loss_2nd +loss_xxx
for i in range(config.loops-1):
    H = encoder(X_standard*(1-config.gain)+X_reconstruct*config.gain,config.dropout,config.is_training)
    X_reconstruct = decoder(H,config.dropout,config.is_training)

    loss_2nd = get_2nd_loss(X_standard, X_reconstruct, config.beta)
    loss_1st = get_1st_loss(H, adjacent_matriX,config.beta_l)
    loss_xxx = tf.reduce_sum(tf.pow(X_reconstruct,2))
    d_loss = d_loss*config.loss_gain+(1-config.loss_gain)*(config.gamma * loss_1st + config.alpha * loss_2nd +loss_xxx)

# no dropout inference
H0 = encoder(X_standard)
X_reconstruct0 = decoder(H0)
for i in range(config.loops-1):
    H0 = encoder(X_standard*(1-config.gain)+X_reconstruct0*config.gain)
    X_reconstruct0 = decoder(H0)

# feature space constraint/generator
H_distorted = H_ref+tf.random_normal(tf.shape(H_ref),stddev = config.g_stddev)
X_reconstruct_on_H = decoder(H_distorted,config.dropout,config.is_training)
H_reconstruct = encoder(X_reconstruct_on_H,config.dropout,config.is_training)
for i in range(config.loops-1):
    X_reconstruct_on_H = decoder(H_distorted*(1-config.gain)+H_reconstruct*config.gain,config.dropout,config.is_training)
    H_reconstruct = encoder(X_reconstruct_on_H,config.dropout,config.is_training)
g_loss = get_g_loss(H_reconstruct,H_distorted,X_reconstruct_on_H,X_standard,config.beta)

###############################################
############### define optimizer ##############
d_opt = tf.train.RMSPropOptimizer(config.learning_rate).minimize(d_loss)
g_opt = tf.train.RMSPropOptimizer(config.g_learning_rate).minimize(g_loss)
# g_opts = tf.train.RMSPropOptimizer(config.g_learning_rate).minimize(g_loss_cumul)
###############################################
############### initialization ################

sess.run(tf.global_variables_initializer())


# In[4]:

# In[5]:


def check_multi_label_f1(label_data,test_ratio = 0.2):
    delete_list = []
    pos = 0
    for i in label_data['label']:
        if i[0]>=np.sum(i):
            delete_list.append(pos)
        pos += 1

    X = sess.run(H0,feed_dict = {X_standard:label_data['adjacent_matriX']})
    Y = label_data['label']
    X = np.delete(X,delete_list,0)
    Y = np.delete(Y,delete_list,0)
    x_train, x_test, y_train, y_test = train_test_split(X, Y[:,1:], test_size = test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred_0 = clf.predict(x_test)
    micro = f1_score(y_test, y_pred_0, average = "micro")
    macro = f1_score(y_test, y_pred_0, average = "macro")
    print("micro_f1: %.4f" % (micro))
    print("macro_f1: %.4f" % (macro))


# In[6]:


config.epochs_limit = 500
epoch = 0
time_consumed = 0
activate_gopt = 3000
# activate_gopt = -1

new_links = copy.deepcopy(graph.links)
new_adj = copy.deepcopy(graph.adj_matrix)
hide_ratio = 0.8
cur_l = len(graph.links)
cur_iter = 0
while(cur_l>int(hide_ratio*origin_data.E)):
    candidate = np.random.randint(0,cur_l)
    cur_link = new_links[candidate]
#         break
    if (np.sum(new_adj[cur_link[0]])>1) & (np.sum(new_adj[cur_link[1]])>1):
        cur_l += -1
        new_links.pop(candidate)
        new_adj[cur_link[0],cur_link[1]] = 0
        new_adj[cur_link[1],cur_link[0]] = 0
    if cur_iter == 5000000:
        break
    cur_iter += 1
graph.links = new_links
graph.adj_matrix = new_adj
graph.E = cur_l

while(epoch<=config.epochs_limit):
    epoch += 1
    graph.order = np.sort(list(graph.order))
    np.random.shuffle(graph.order)
    epoch_not_end = True
    pos = 0
    loss = 0

    while(epoch_not_end):
        epoch_run_time = time.time()

        minibatch = graph.adj_matrix[graph.order[pos:np.min([pos+config.batch_size,graph.N])]]
        minibatch_adj = graph.adj_matrix[graph.order[pos:np.min([pos+config.batch_size,graph.N])]][:,graph.order[pos:np.min([pos+config.batch_size,graph.N])]]
        minibatch_z = np.zeros([np.min([pos+config.batch_size,graph.N])-pos,config.struct[-1]])

        # update encoder
        _,H_temp = sess.run([d_opt,H], feed_dict = {X_standard:minibatch,adjacent_matriX:minibatch_adj})
        # update decoder
        if (epoch>activate_gopt):
            for i_d in range(int(np.min([np.round((epoch//150)**2),config.loops_H]))):
                _ = sess.run(g_opt, feed_dict = {X_standard:minibatch,H_ref:H_temp})

        time_consumed += time.time()-epoch_run_time

        if pos+config.batch_size>=graph.N:
            epoch_not_end = False
        else:
            pos += config.batch_size
    if epoch%config.display == 0:
        graph.order = np.sort(list(graph.order))
        [bbc,loss1] = sess.run([H0,d_loss],feed_dict={X_standard:graph.adj_matrix,adjacent_matriX:graph.adj_matrix})
        loss2 = sess.run(g_loss,feed_dict = {X_standard:origin_data.adj_matrix,H_ref:bbc})
        print("epoch: %d,\td_loss:%.3f,\tg_loss:%.3f,\ttime:%.3fs" % (epoch, loss1,loss2,time_consumed))
        check_link_prediction(bbc,graph, origin_data,  [int((1-hide_ratio)*420889*i) for i in [0.1,0.2,0.4,0.6,0.8, 1]])
        label_data = origin_data.sample(origin_data.N, with_label = True)
