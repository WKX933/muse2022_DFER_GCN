from torch.nn import Parameter
import math
from GCN_getA import *
import torch.nn as nn
# from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from config import ACTIVATION_FUNCTIONS
from Former_DFER_main.models.pre_ST_GCN import GenerateModel
# from linformer_pytorch import Linformer
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


#将model.py中的GCNResnet类改成我自己的GCNDensenet类
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Model(nn.Module):
    #in_channel是指词向量的维度，即一个词由300维的向量表示，t表示阈值，adj_file是我们上面生成adj_file的文件地址
    def __init__(self,  num_classes=7, in_channel=300, t=0.2, adj_file='/data6/wangkexin/muse2022/adj.pkl'):
        super(Model, self).__init__()
        self.model = GenerateModel()
        self.num_classes = num_classes
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.gc1 = GraphConvolution(in_channel, 512)
        #self.gc2 = GraphConvolution(1024, 2048)  因为我densenet-121最后的feature是1024维，所以把这个地方的2048改成了1024
        self.gc2 = GraphConvolution(512, 512)
        self.relu = nn.LeakyReLU(0.2)
        #获取A矩阵
        _adj = gen_A(num_classes,t, adj_file)
        # print("邻接矩阵为：",_adj)
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.final_activation = torch.nn.Sigmoid()
        # image normalization  由于我的数据都已经归一化过了，所以这部分就没有用

    # feature就是CNN网络的输入，inp就是word embedding
    def forward(self, feature, inp='/data6/wangkexin/muse2022/glove_wordEmbedding.pkl'):
        # Densenet feature extract
        feature = self.model(feature)
        # feature = self.pooling(feature)
        # feature = feature.view(feature.size(0), -1)
        #the shape of feature is (batch_size, 1024)

        # 2层的GCN网络
        # word embedding
        adj = gen_adj(self.A).detach()
        inp = pickle.load(open(inp, 'rb'))
        inp = torch.from_numpy(inp).cuda()
        inp= inp.float()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        #the shape of x is (类别数, 1024)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return self.final_activation(x)
