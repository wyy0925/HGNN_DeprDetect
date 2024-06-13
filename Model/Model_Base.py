# 1-D CNN EEG Temporal Feature Extraction
import torch
from torch import nn,einsum
from torch.nn import functional as F
from einops import rearrange, repeat


class EEGEncoder(nn.Module):
    def __init__(self, drop_out=0.25, b1_kernel=63, b1_F1=16, b1_pool_kernel=4, encoder_out=32, input_sp=1000):
        super(EEGEncoder, self).__init__()
        self.drop_out = drop_out

        self.F1 = b1_F1
        self.F2 = encoder_out
        self.b1_kernel = b1_kernel
        self.b1_pool_kernel = b1_pool_kernel
        self.b1_pad = int((self.b1_kernel - 1) / 2)
        self.b3_kernel = int(input_sp / self.b1_pool_kernel)

        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.F1, kernel_size=self.b1_kernel, bias=False, padding=self.b1_pad),
            nn.BatchNorm1d(self.F1),
            nn.ELU(),
            nn.AvgPool1d(self.b1_pool_kernel),
            nn.Dropout(self.drop_out)
        )

        self.block_2 = nn.Sequential(
            nn.Conv1d(in_channels=self.F1, out_channels=self.F2, kernel_size=1),
            nn.BatchNorm1d(self.F2),
            nn.ELU()
        )

        self.block_3 = nn.Sequential(
            nn.Conv1d(in_channels=self.F2, out_channels=self.F2, kernel_size=self.b3_kernel, groups=self.F2, bias=False,
                      padding=0),
            nn.BatchNorm1d(self.F2),
            nn.ELU(),
            nn.Dropout(self.drop_out)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return x  # return x for visualization


def normalize_A(A):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    A = A + torch.eye(A.shape[0], device=A.device)
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    Lnorm=torch.matmul(torch.matmul(D, A), D)
    return Lnorm


def BatchAdjNorm(A):
    bs,h, n, _ = A.shape
    # A = A.reshape(bs * h, n, n)
    A = F.relu(A)
    identity = torch.eye(n, n,device=A.device)
    identity_matrix = identity.repeat(bs,h, 1, 1)
    A = A * (torch.ones(bs,h, n, n,device=A.device) - identity_matrix)
    A = A + A.transpose(2, 3)
    A = A + identity_matrix
    d = torch.sum(A, 3)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    Lnorm = torch.matmul(torch.matmul(D, A), D)
    return Lnorm

class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, head=4,bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda())
        self.head = head
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = repeat(x, 'b n d -> b h n d', h=self.head)
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


def generate_adj(L, K):
    support = []
    L_iter=L
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        else:
            support.append(L_iter)
            L_iter=L_iter*L
    return support


class GNNnet(nn.Module):
    def __init__(self, in_channels, out_channels,head,K):
        super(GNNnet, self).__init__()
        self.K = K
        self.gnn = nn.ModuleList()
        for i in range(self.K):
            self.gnn.append(GraphConvolution(in_channels, out_channels, head))

    def forward(self, x,L):
        adj = generate_adj(L, self.K)
        for i in range(len(self.gnn)):
            if i == 0:
                result = F.leaky_relu(self.gnn[i](x, adj[i]))
            else:
                result += F.leaky_relu(self.gnn[i](x, adj[i]))
        return result



class AdjGenerator(nn.Module):
    def __init__(self, dim,head, Tem=1, dim_head=64):
        super().__init__()
        self.head=head
        inner_dim = dim_head * self.head
        self.scale = Tem ** -1
        self.attend = nn.Softmax(dim=-1)
        self.to_qk = nn.Linear(dim, inner_dim * self.head, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.head), qk)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        attn = BatchAdjNorm(attn)
        return attn

# GCNN Constructed based on https://github.com/xueyunlong12589/DGCNN  DGCNN
class GCNN(nn.Module):

    def __init__(self, in_channels, out_channels,k,head):
        super(GCNN, self).__init__()
        self.K = k
        self.layer1 = GNNnet(in_channels, out_channels, head,self.K)
        self.BN1 = nn.BatchNorm2d(1)

    def forward(self, x,A):
        x = self.BN1(x.unsqueeze(1)).squeeze(1)  # data can also be standardized offline
        x = self.layer1(x, A)
        return x


class Pool_Generator(nn.Module):
    def __init__(self, GNN_in, GNN_out_dim, N_next, head=1, k=1, namda=0.5):
        super(Pool_Generator, self).__init__()
        self.para = namda
        self.R_gene = GCNN(GNN_in, N_next, k, head)
        self.gnn = GCNN(GNN_in, GNN_out_dim, k, head)

    def forward(self, x, L):
        R = F.softmax(self.R_gene(x, L), dim=-1).squeeze(1)
        Ar = torch.matmul(torch.matmul((R.transpose(-1, -2)), L.squeeze(1)), R)
        Ar = BatchAdjNorm(Ar.unsqueeze(1))
        X_pool = torch.matmul(R.transpose(1, 2), x)
        X_pool = self.gnn(X_pool, Ar).squeeze(1)
        X_unpool = torch.matmul(R, X_pool)
        x_out = X_unpool * self.para + x
        return x_out, R