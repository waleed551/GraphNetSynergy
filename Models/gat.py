import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=64, dropout=0.2):
        super(GATNet, self).__init__()

        # graph layers for drug1
        self.D1_gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.D1_gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.D1_fc_g1 = nn.Linear(output_dim, output_dim)
        
        # graph layers for drug2
        self.D2_gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.D2_gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.D2_fc_g1 = nn.Linear(output_dim, output_dim)

        # cell line gene expression
        self.fc1_xt = nn.Linear(1000,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.out = nn.Linear(32, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data1,data2):
        # graph input feed-forward for drug1
        x1, edge_index_1, batch1 = data1.x, data1.edge_index, data1.batch
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = F.elu(self.D1_gcn1(x1, edge_index_1))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.D1_gcn2(x1, edge_index_1)
        x1 = self.relu(x1)
        # global max pooling
        x1 = gmp(x1, batch1)          
        x1 = self.D1_fc_g1(x1)
        x1 = self.relu(x1)
        
        # graph input feed-forward for drug2
        x2, edge_index_2, batch2 = data2.x, data2.edge_index, data2.batch
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = F.elu(self.D2_gcn1(x2, edge_index_2))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = self.D2_gcn2(x2, edge_index_2)
        x2 = self.relu(x2)
        # global max pooling
        x2 = gmp(x2, batch2)          
        x2 = self.D2_fc_g1(x2)
        x2 = self.relu(x2)
        
        # flatten
        xd = torch.cat((x1, x2), 1)

        # cell line expression input feed-forward:
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
