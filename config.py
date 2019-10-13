class Config(object):
    def __init__(self,data_graph,data_graph_label,data_sequence):
        ## graph data
        self.data_graph = 'data/'+data_graph
        self.label_file_path = 'data/'+data_graph_label
        self.sequence_path = 'data/'+data_sequence
        ## embedding data
        self.embedding_filename = 'results/'+data_graph
        ## hyperparameter
        self.struct = [None, 200]
        self.seq_struct = [None,200]
        self.gamma = 1
        self.alpha_seq = 1
        self.reg = 1
        self.g_lambda = 500
        self.g_lambda0 = 1
        self.g_lambda_seq = 5000
        self.g_regX = 1
        self.g_regH = 1
        self.g_learning_rate = 0.0001
        self.beta = 10
        self.beta_l = 0.0
        
        self.dropout = [0.0,0.0]
        self.is_training = True
        self.loss_gain = 0.1
        self.loops = 1
        self.loops_H = 1
        self.gain = 0.1
        self.g_stddev = 1e-4
        
        ## para for training
        self.batch_size = 100
        self.epochs_limit = 1000
        self.learning_rate = 0.001
        self.display = 10

        self.DBN_init = True
        self.sparse_dot = False
        self.ng_sample_ratio = 0.0 
        self.sample_ratio = 1
        self.sample_method = "node"
    
    def __str__(self):
        return "parameter: struct %s, learning_rate %s, dropout %s, loops %s, loss_gain %s, gain %s" % (self.struct,self.learning_rate,self.dropout,self.loops,self.loss_gain,self.gain)
