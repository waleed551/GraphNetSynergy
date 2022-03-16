import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
############################### GCN based model ##############################
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd1=78,num_features_xd2=78, num_features_xt=1000, output_dim=64, dropout=0.2):

        super(GCNNet, self).__init__()

        # Drug1 SMILES graph branch
        self.n_output = n_output
        self.D1_conv1 = GCNConv(num_features_xd1, num_features_xd1)
        self.D1_conv2 = GCNConv(num_features_xd1, num_features_xd1*2)
        self.D1_conv3 = GCNConv(num_features_xd1*2, num_features_xd1 * 4)
        self.D1_fc_g1 = torch.nn.Linear(num_features_xd1*4, 1024)
        self.D1_fc_g2 = torch.nn.Linear(1024, output_dim)
        self.D1_relu = nn.ReLU()
        self.D1_dropout = nn.Dropout(dropout)
        
        # Drug2 SMILES graph branch
        self.D2_conv1 = GCNConv(num_features_xd2, num_features_xd2)
        self.D2_conv2 = GCNConv(num_features_xd2, num_features_xd2*2)
        self.D2_conv3 = GCNConv(num_features_xd2*2, num_features_xd2 * 4)
        self.D2_fc_g1 = torch.nn.Linear(num_features_xd2*4, 1024)
        self.D2_fc_g2 = torch.nn.Linear(1024, output_dim)
        self.D2_relu = nn.ReLU()
        self.D2_dropout = nn.Dropout(dropout)

        # cell line gene expression 
        self.fc1_xt = nn.Linear(1000,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.out = nn.Linear(32, self.n_output)

    def forward(self, data1,data2):
        # get graph input for Drug1 
        x1, edge_index_1, batch1 = data1.x, data1.edge_index, data1.batch
        
        x1 = self.D1_conv1(x1, edge_index_1)
        x1 = self.D1_relu(x1)

        x1 = self.D1_conv2(x1, edge_index_1)
        x1 = self.D1_relu(x1)

        x1 = self.D1_conv3(x1, edge_index_1)
        x1 = self.D1_relu(x1)
        # global max pooling
        x1 = gmp(x1, batch1)       

        # flatten
        x1 = self.D1_relu(self.D1_fc_g1(x1))
        x1 = self.D1_dropout(x1)
        x1 = self.D1_fc_g2(x1)
        x1 = self.D1_dropout(x1)
        
        # get graph input for Drug2
        x2, edge_index_2, batch2 = data2.x, data2.edge_index, data2.batch
        
        x2 = self.D2_conv1(x2, edge_index_2)
        x2 = self.D2_relu(x2)

        x2 = self.D2_conv2(x2, edge_index_2)
        x2 = self.D2_relu(x2)

        x2 = self.D2_conv3(x2, edge_index_2)
        x2 = self.D2_relu(x2)
        # global max pooling
        x2 = gmp(x2, batch2)       
        
        # flatten
        x2 = self.D2_relu(self.D2_fc_g1(x2))
        x2 = self.D2_dropout(x2)
        x2 = self.D2_fc_g2(x2)
        x2 = self.D2_dropout(x2)
        xd = torch.cat((x1, x2), 1)

        # get cell line expression as input
        xt = data1.target
        xt=xt.view(-1,1000)
        xt = self.fc1_xt(xt)
        
        # concat
        xc = torch.cat((xd, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out