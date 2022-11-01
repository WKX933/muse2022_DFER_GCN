import torch
from torch import nn
from Former_DFER_main.models.S_Former import spatial_transformer
from Former_DFER_main.models.T_Former import temporal_transformer

class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y

class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        self.fc = nn.Linear(512, 7)
        self.fc_1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(True))
        self.fc_2 = nn.Linear(128, 7)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):

        x = self.s_former(x)
        x = self.t_former(x)
        x = self.fc_2(self.fc_1(x))
        return self.activate(x)



if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112)) #batch*frames*channels*H*W
    model = GenerateModel()
    model(img)
