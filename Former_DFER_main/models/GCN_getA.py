import torch
import numpy as np
import pickle
import pandas as pd
def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1

    #ps:这个地方好像跟论文里面的公式有出入，但是它代码是这样写的，我也就按照它代码来处理
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

#gen_adj()相当于通过A得到A_hat矩阵
def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def make_adj_file():  ###只要train
    import pandas as pd
    from config import PATH_TO_LABELS,PARTITION_FILES
    import os

    path_par=PARTITION_FILES['reaction']
    parser_df = pd.read_csv(path_par)
    parser = parser_df.values
    label_path = PATH_TO_LABELS['reaction']
    label_df = pd.read_csv(os.path.join(label_path, 'labelOFmulticlassification.csv'))
    dataset = label_df.iloc[:, 1:].values
    for i in range(len(parser_df)):
        if parser[i,1] !=  'train':
            np.delete(dataset,i,0)
    #共现矩阵 shape is (14,14)
    adj_matrix = np.zeros(shape=(7, 7))
    #每个类别出现的总次数 shape is (14, )
    nums_matrix = np.zeros(shape=(7))

    '''
    算法思路
    一、遍历每一行数据
        1、统计每一行中两两标签出现次数（自己和自己的不统计在内，即adj_matrix是一个对称矩阵且对角线为0）
        2、统计每一行中每个类别出现的次数
    '''
    for index in range(len(dataset)):
        data = dataset[index]
        for i in range(7):
            if data[i] == 1:
                nums_matrix[i] += 1
                for j in range(7):
                    if j != i:
                        if data[j] == 1:
                            adj_matrix[i][j] += 1

    adj = {'adj': adj_matrix,
           'nums': nums_matrix}
    pickle.dump(adj, open('./adj.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
