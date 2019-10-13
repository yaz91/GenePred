import numpy as np
from utils import *
import random
import copy
class Graph(object):
    def __init__(self, file_path, ng_sample_ratio):
        suffix = file_path.split('.')[-1]
        self.st = 0
        self.is_epoch_end = False
        if suffix == "txt":
            fin = open(file_path, "r")
            firstLine = fin.readline().strip().split()
            self.N = int(firstLine[0])
            self.E = int(firstLine[1])
            self.__is_epoch_end = False
            self.adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.links = np.zeros([self.E + int(ng_sample_ratio*self.N) , 3], np.int_)
            count = 0
            for line in fin.readlines():
                line = line.strip().split('\t')
                self.adj_matrix[int(line[0]),int(line[1])] += 1
                self.adj_matrix[int(line[1]),int(line[0])] += 1
                self.links[count][0] = int(line[0])
                self.links[count][1] = int(line[1])
                self.links[count][2] = 1
                count += 1
            fin.close()
            if (ng_sample_ratio > 0):
                self.__negativeSample(int(ng_sample_ratio*self.N), count, self.adj_matrix.copy())
            self.order = np.arange(self.N)
            print("getData done")
            print("Vertexes : %d  Edges : %d ngSampleRatio: %f" % (self.N, self.E, ng_sample_ratio))
        else:
            pass
            #TODO read a mat file or something like that.
        
    def __negativeSample(self, ngSample, count, edges):
        print("negative Sampling")
        size = 0
        while (size < ngSample):
            xx = random.randint(0, self.N-1)
            yy = random.randint(0, self.N-1)
            if (xx == yy or edges[xx][yy] != 0):
                continue
            edges[xx][yy] = -1
            edges[yy][xx] = -1
            self.links[size + count] = [xx, yy, -1]
            size += 1
        print("negative Sampling done")
        
    def load_label_data(self, filename):
        with open(filename,"r") as fin:
            firstLine = fin.readline().strip().split()
            self.label = np.zeros([self.N, int(firstLine[1])], np.bool)
            lines = fin.readlines()
            for line in lines:
                line = line.strip().split(' : ')
                if len(line) > 1:
                    labels = line[1].split()
                    for label in labels:
                        self.label[int(line[0])][int(label)] = True
        self.has_label = np.ones([self.N,1])
        j = 0
        for i in self.label:
            if i[0]>=np.sum(i):
                self.has_label[j] = 0
            j += 1
    
    def save_graph(self,filename):
        with open(filename,"w") as fout:
            fout.writelines([str(self.N),'\t',str(self.E),'\n'])
            [fout.writelines([str(self.links[i][0]),'\t',str(self.links[i][1]),'\n']) for i in range(self.E)]    
    
    def load_sequence_data(self, filename):
        self.sequence_feature = []
        with open(filename,"r") as fin:
            firstLine = fin.readline().strip().split()
            self.seq_feature_num = int(firstLine[1])
            for line in fin:
                self.sequence_feature.append([float(i) for i in line.strip().split('\t')][1:])
        self.sequence_feature = np.asarray(self.sequence_feature)
                        
    def sample(self, batch_size, do_shuffle = True, with_label = False):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(list(self.order)[0:self.N])
            else:
                self.order = list(np.sort(list(self.order)))
            self.st = 0
            self.is_epoch_end = False 
        mini_batch = Dotdict()
        en = min(self.N, self.st + batch_size)
        index = list(self.order)[self.st:en]     
        mini_batch.X = self.adj_matrix[index]
        mini_batch.adjacent_matriX = self.adj_matrix[index][:,index]
        mini_batch.sequence_feature = self.sequence_feature
        if with_label:
            mini_batch.label = self.label[index]
        if (en == self.N):
            en = 0
            self.is_epoch_end = True
        self.st = en
        return mini_batch
    
    def subgraph(self, method, sample_ratio):
        new_N = int(sample_ratio * self.N)
        cur_N = 0
        if method == "link":
            new_links = []
            self.adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.order = {}
            while (cur_N < new_N):
                p = int(random.random() * self.E)
                link = self.links[p]
                if self.adj_matrix[link[0]][link[1]] == 0:
                    new_links.append(link)
                    self.adj_matrix[link[0]][link[1]] = 1
                    self.adj_matrix[link[1]][link[0]] = 1
                    if link[0] not in self.order:
                        self.order[link[0]] = 1
                        cur_N += 1
                    if link[1] not in self.order:
                        self.order[link[1]] = 1
                        cur_N += 1
            self.links = new_links
            self.order = self.order.keys()
            self.N = new_N
            print(len(self.links))
            return self
        elif method == "node":
            self.adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.order = {}
            new_links = []
            while (cur_N < new_N):
                p = int(random.random() * self.N)
                if p not in self.order:
                    self.order[p] = 1
                    cur_N += 1
            for link in self.links:
                if link[0] in self.order and link[1] in self.order:
                    self.adj_matrix[link[0]][link[1]] = 1
                    self.adj_matrix[link[1]][link[0]] = 1
                    new_links.append(link)
            self.order = self.order.keys()
            self.N = new_N
            self.links = new_links
            print(len(self.links))
            return self
            pass
        elif method == "explore": 
            new_adj_matrix = np.zeros([self.N, self.N], np.int_)
            self.order = {}
            new_links = []
            while (cur_N < new_N):
                p = int(random.random() * self.N)
                k = int(random.random() * 100)
                for i in range(k):
                    if p not in self.order:
                        self.order[p] = 1
                        cur_N += 1
                    b = self.adj_matrix[p].nonzero()
                    b = b[0]
                    w = int(random.random() * len(b))
                    new_adj_matrix[p][b[w]] = 1
                    new_adj_matrix[b[w]][p] = 1
                    new_links.append([p,b[w],1])
                    p = b[w]
            self.order = self.order.keys()
            self.adj_matrix = new_adj_matrix
            self.N = new_N
            self.links = new_links
            print(len(self.links))
            return self
            pass
    
