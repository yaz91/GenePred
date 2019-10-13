import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import pdb

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def getSimilarity(result):
    print("getting similarity...")
    return np.dot(result, result.T)-np.diag(np.diag(np.dot(result, result.T)))
    
def check_link_reconstruction(embedding, graph_data, check_index,print_to_file = False, fout = None):
    def get_precisionK(embedding, data, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind // data.N
            y = ind % data.N
            count += 1
            if (data.adj_matrix[x][y] == 1):# or x == y):
                cur += 1 
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    for index in check_index:
        if print_to_file:
            print("precisonK[%d] %.2f" % (index, precisionK[index - 1]), file = fout)
        else:
            print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
            
def check_link_prediction(embedding, graph_train,graph_test, check_index,print_to_file = False, fout = None):
    def get_precisionK(embedding, data_train,data_test, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind // data_train.N
            y = ind % data_train.N
            if (data_train.adj_matrix[x][y] == 1):
                next
            elif (data_test.adj_matrix[x][y] == 1):
                cur += 1
                count += 1
            else:
                count += 1
            precisionK.append(1.0 * cur / np.max([count,1]))
            if count > max_index:
                break
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_train,graph_test, np.max(check_index))
    for index in check_index:
        if print_to_file:
            print("precisonK[%d] %.2f" % (index, precisionK[index - 1]), file = fout)
        else:
            print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
            
def check_link_false_prediction(embedding, graph_train,graph_test, check_index,print_to_file = False, fout = None):
    def get_precisionK(embedding, data_train,data_test, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        idx = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind // data_train.N
            y = ind % data_train.N
            if (data_train.adj_matrix[x][y] == 1):
                next
            elif (data_test.adj_matrix[x][y] == 1):
                cur += 1
                count += 1
            else:
                count += 1
            if cur == check_index[idx]-1:
                precisionK.append(1.0 * cur / np.max([count,1]))
                idx += 1
                if idx == len(check_index):
                    break
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_train,graph_test, np.max(check_index))
    for index in range(len(check_index)):
        if print_to_file:
            print("precisonK[%d] %.2f" % (index, precisionK[index]), file = fout)
        else:
            print("precisonK[%d] %.2f" % (index, precisionK[index]))

def check_multi_label_classification(X, Y, test_ratio = 0.9):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape,np.bool)
        sort_index = np.flip(np.argsort(y_pred, axis = 1), 1)
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    #y_pred = clf.predict_proba(x_test)
    
    ## small trick : we assume that we know how many label to predict
    # y_pred = small_trick(y_test, y_pred)
    
    y_pred = clf.predict(x_test)
    micro = f1_score(y_test[1:], y_pred[1:], average = "micro")
    macro = f1_score(y_test[1:], y_pred[1:], average = "macro")
    print("micro_f1: %.4f" % (micro))
    print("macro_f1: %.4f" % (macro))
    #############################################


    
