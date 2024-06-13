import torch
from torch import nn

from Models.Model_Base import EEGEncoder, Pool_Generator, GCNN, AdjGenerator, normalize_A

# Integrate Graph Pooling and Unpooling Module(GPUM) within FGNN
class HybGNN_Fix(nn.Module):
    def __init__(self, input_sample,GNN_in=32, head=1,N_next=4, Adj_Tem=1, GNN_out_dim=32, adj_dim=64, num_classes=2, chan_num=19):
        super(HybGNN_Fix, self).__init__()
        self.Encoder=EEGEncoder(input_sp=input_sample)
        self.pool_gene=Pool_Generator(GNN_out_dim,GNN_out_dim,N_next)
        self.IAGNN = GCNN(GNN_in, GNN_out_dim, 3, head)
        self.FGNN = GCNN(GNN_in, GNN_out_dim, 2, 1)
        self.A_FGNN = nn.Parameter(torch.FloatTensor(chan_num, chan_num))
        nn.init.uniform_(self.A_FGNN, 0.01, 0.5)
        self.A_IAGNN_Generator = AdjGenerator(GNN_in, head, Adj_Tem, adj_dim)
        self.BN2 = nn.BatchNorm1d(num_classes)
        self.fc = nn.Linear(chan_num * GNN_out_dim * (head + 1), num_classes)

    def forward(self, x):
        bs, chan, _ = x.shape
        x = x.reshape(bs * chan, 1, -1)
        x = self.Encoder(x)
        x = x.reshape(bs, chan, -1)

        # FGNN with GPUM
        L = normalize_A(self.A_FGNN)
        x_fix = self.FGNN(x, L).squeeze(1)
        x_fix,R=self.pool_gene(x_fix,L)

        # IAGNN without GPUM
        A_IAGNN = self.A_IAGNN_Generator(x)
        x_instance = self.IAGNN(x, A_IAGNN).squeeze(1)

        # Concat and Classification
        x_out = torch.concat((x_instance, x_fix), dim=-1).reshape((bs, -1))
        x_out = self.fc(x_out)
        x_out = self.BN2(x_out)
        return x_out, R



# Integrate Graph Pooling and Unpooling Module(GPUM) within IAGNN  (Our proposed model)
class HybGNN_IA(nn.Module):
    def __init__(self, input_sample,GNN_in=32, head=1,N_next=4, Adj_Tem=1, GNN_out_dim=32, adj_dim=64, num_classes=2, chan_num=19):
        super(HybGNN_IA, self).__init__()
        self.Encoder = EEGEncoder(input_sp=input_sample)
        self.pool_gene=Pool_Generator(GNN_out_dim,GNN_out_dim,N_next)
        self.IAGNN = GCNN(GNN_in, GNN_out_dim, 2, head)
        self.FGNN = GCNN(GNN_in, GNN_out_dim, 3, 1)
        self.A_FGNN = nn.Parameter(torch.FloatTensor(chan_num, chan_num))
        nn.init.uniform_(self.A_FGNN, 0.01, 0.5)
        self.A_IAGNN_Generator = AdjGenerator(GNN_in, head, Adj_Tem, adj_dim)
        self.BN2 = nn.BatchNorm1d(num_classes)
        self.fc = nn.Linear(chan_num * GNN_out_dim * (head + 1), num_classes)

    def forward(self, x):
        bs,chan,_=x.shape
        x = x.reshape(bs * chan, 1, -1)
        x = self.Encoder(x)
        x = x.reshape(bs, chan, -1)

        # FGNN without GPUM
        L = normalize_A(self.A_FGNN)
        x_fix = self.FGNN(x, L).squeeze(1)

        # IAGNN with GPUM
        A_IAGNN = self.A_IAGNN_Generator(x)
        x_instance = self.IAGNN(x, A_IAGNN).squeeze(1)
        x_instance, R = self.pool_gene(x_instance, A_IAGNN)

        x_out = torch.concat((x_instance, x_fix), dim=-1).reshape((bs,-1))
        x_out = self.fc(x_out)
        x_out = self.BN2(x_out)
        return x_out, R, A_IAGNN


if __name__ == '__main__':
    # for MODMA
    data = torch.rand((128, 19, 1000)).cuda()
    model = HybGNN_IA(input_sample=1000).cuda()
    out = model(data)
    # for HUSM
    data_ = torch.rand((128, 19, 1024)).cuda()
    model_ = HybGNN_IA(input_sample=1024).cuda()
    out_ = model_(data_)
