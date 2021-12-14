#from dgl.nn.pytorch.conv import GINConv
from unicodedata import bidirectional
from Module import GINConv
import torch.nn as nn
import dgl
from dgllife.model.readout import WeightedSumAndMax, AttentiveFPReadout, WeaveGather
import torch
from rdkit.Chem.Descriptors import *
import numpy as np
import rdkit.Chem as Chem
from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer,RobustScaler
from torch.nn.functional import affine_grid
class MyGIN(nn.Module):
    def __init__(self, in_feat, out_feat, num_classes, p, num_layers, apply_func = None, aggregator_type=None, init_eps = 1e-5, learn_eps = False):
        super(MyGIN, self).__init__()
        self.descriptors = [ExactMolWt, HeavyAtomMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3,
                            MaxAbsPartialCharge, MaxPartialCharge, MinAbsPartialCharge, MinPartialCharge,
                            MolWt, NumRadicalElectrons, NumValenceElectrons]
        if apply_func is None:
            # self.apply_funcs = nn.ModuleList()
            # for i in range(num_layers):
            #     self.apply_funcs.append(nn.Linear(in_features=out_feat, out_features=out_feat))
            #     self.apply_funcs.append(nn.GRUCell(input_size=out_feat,hidden_size=out_feat, bias=True))
            self.apply_func = nn.GRUCell(input_size=out_feat * 2, hidden_size=out_feat, bias=True)
            #self.apply_func = nn.Linear(in_features=out_feat, out_features=out_feat)
        else:
            self.apply_func = apply_func
        
        if aggregator_type is None:
            aggregator_type = 'sum'
        self.convs = nn.ModuleList()
        self.batchNorms = nn.ModuleList()
        self.layerNorms = nn.ModuleList()
        self.dropout = nn.Dropout(p=p)
        for i in range(num_layers):
            self.convs.append(GINConv(apply_func= self.apply_func, aggregator_type=aggregator_type, hidden_size=out_feat, init_eps=init_eps, learn_eps=learn_eps))
            self.batchNorms.append(nn.BatchNorm1d(num_features=out_feat))
            self.layerNorms.append(nn.LayerNorm(out_feat, eps = 1e-5, elementwise_affine = True))
        
        self.readout = WeightedSumAndMax(out_feat)
        
        self.input = nn.Linear(in_features=in_feat, out_features=out_feat)
        #self.out1 = nn.Linear(in_features=out_feat * (num_layers + 1) * 2 * 2,  out_features=num_classes)
        self.out1 = nn.Linear(in_features=out_feat * 2,  out_features=num_classes)
        self.drop1 = nn.Dropout(p = 0.2)
        self.drop2 = nn.Dropout(p = 0.3)
        self.drop3 = nn.Dropout(p = 0.4)
        self.lstm = nn.LSTM(input_size = out_feat * 2, hidden_size = out_feat, num_layers = 1, bidirectional=True)
        self.init()
    
    def init(self):
        
        gain = nn.init.calculate_gain('relu')
        
        nn.init.kaiming_normal_(self.apply_func.weight_hh)
        nn.init.kaiming_normal_(self.apply_func.weight_ih)
        #nn.init.kaiming_normal_(self.apply_func.bias_hh)
        #nn.init.kaiming_normal_(self.apply_func.bias_ih)
        # nn.init.xavier_normal_(self.apply_func.weight, gain=gain)
        # nn.init.xavier_normal_(self.apply_func.bias, gain=gain)
        
    
    def forward(self, g, feats, smiles):
        feats = self.input(feats)
        graph_representation = []
        #graph_representation.append(self.readout(g, feats))
        for conv, batchNorm in zip(self.convs, self.batchNorms):
            feat = batchNorm(feats)
            
            if self.training:
                feat = self.dropout(feat)
            feat = conv(g, feat)
            feat = torch.nn.functional.celu(feat)
            graph_representation_t = self.readout(g, feats)
            # resdiuals
            feats = feat + feats
            
            graph_representation.append(graph_representation_t)
        # 将graph representation经过一轮Lstm进行拟合，然后取第一个
        graph_representation = torch.stack(graph_representation, dim = 0)
        graph_representation = self.lstm(graph_representation)[0][0]
        

        if self.training:
            graph_representation1 = self.drop1(graph_representation)
            graph_representation2 = self.drop2(graph_representation)
            graph_representation3 = self.drop3(graph_representation)
            

            logit1, logit2, logit3 = self.out1(graph_representation1), self.out1(graph_representation2), self.out1(graph_representation3)
            
            return logit1, logit2, logit3
        else:
            feats = self.out1(graph_representation)
            return feats

    def encode_graph_repr(self, smiles):
        matrix = np.zeros((len(smiles), len(self.descriptors)), dtype=np.float32)
        for i in range(len(smiles)):
            mol = Chem.MolFromSmiles(smiles[i])
            for j in range(len(self.descriptors)):
                matrix[i][j] = self.descriptors[j](mol)
        matrix = RobustScaler().fit_transform(matrix)
        matrix = torch.from_numpy(matrix)
        return matrix


if __name__=="__main__":
    g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    feat = torch.ones(6, 10)
    lin = nn.Linear(10, 10)
    gin = MyGIN(10, 10, 2, 0.2, 2, lin, 'max')
    feat = gin(g, feat)
    print(feat)
    pass

