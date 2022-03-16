import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=64, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers for drug1
        D1_nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.D1_conv1 = GINConv(D1_nn1)
        self.D1_bn1 = torch.nn.BatchNorm1d(dim)

        D1_nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.D1_conv2 = GINConv(D1_nn2)
        self.D1_bn2 = torch.nn.BatchNorm1d(dim)

        D1_nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.D1_conv3 = GINConv(D1_nn3)
        self.D1_bn3 = torch.nn.BatchNorm1d(dim)

        D1_nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.D1_conv4 = GINConv(D1_nn4)
        self.D1_bn4 = torch.nn.BatchNorm1d(dim)

        D1_nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.D1_conv5 = GINConv(D1_nn5)
        self.D1_bn5 = torch.nn.BatchNorm1d(dim)

        self.D1_fc1_xd = Linear(dim, output_dim)
        
        # convolution layers for drug2
        D2_nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.D2_conv1 = GINConv(D2_nn1)
        self.D2_bn1 = torch.nn.BatchNorm1d(dim)

        D2_nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.D2_conv2 = GINConv(D2_nn2)
        self.D2_bn2 = torch.nn.BatchNorm1d(dim)

        D2_nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.D2_conv3 = GINConv(D2_nn3)
        self.D2_bn3 = torch.nn.BatchNorm1d(dim)

        D2_nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.D2_conv4 = GINConv(D2_nn4)
        self.D2_bn4 = torch.nn.BatchNorm1d(dim)

        D2_nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.D2_conv5 = GINConv(D2_nn5)
        self.D2_bn5 = torch.nn.BatchNorm1d(dim)

        self.D2_fc1_xd = Linear(dim, output_dim)

        # feature from gene expression
        self.fc1_xt = nn.Linear(1000, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.out = nn.Linear(32, self.n_output)        

    def forward(self, data1,data2):
        x1, edge_index_1, batch1 = data1.x, data1.edge_index, data1.batch
        #For forward drug1
        x1 = F.relu(self.D1_conv1(x1, edge_index_1))
        x1 = self.D1_bn1(x1)
        x1 = F.relu(self.D1_conv2(x1, edge_index_1))
        x1 = self.D1_bn2(x1)
        x1 = F.relu(self.D1_conv3(x1, edge_index_1))
        x1 = self.D1_bn3(x1)
        x1 = F.relu(self.D1_conv4(x1, edge_index_1))
        x1 = self.D1_bn4(x1)
        x1 = F.relu(self.D1_conv5(x1, edge_index_1))
        x1 = self.D1_bn5(x1)
        x1 = global_add_pool(x1, batch1)
        x1 = F.relu(self.D1_fc1_xd(x1))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        x2, edge_index_2, batch2 = data2.x, data2.edge_index, data2.batch
        #For forward drug1
        x2 = F.relu(self.D2_conv1(x2, edge_index_2))
        x2 = self.D2_bn1(x2)
        x2 = F.relu(self.D2_conv2(x2, edge_index_2))
        x2 = self.D2_bn2(x2)
        x2 = F.relu(self.D2_conv3(x2, edge_index_2))
        x2 = self.D2_bn3(x2)
        x2 = F.relu(self.D2_conv4(x2, edge_index_2))
        x2 = self.D2_bn4(x2)
        x2 = F.relu(self.D2_conv5(x2, edge_index_2))
        x2 = self.D2_bn5(x2)
        x2 = global_add_pool(x2, batch2)
        x2 = F.relu(self.D2_fc1_xd(x2))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        # flatten the drugs features
        xd = torch.cat((x1, x2), 1)
        
        # get gene expression input
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
