import torch
from torch import nn
import torch.nn.functional as F

class identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    def forward(self, X):
        return X

class SKConv(nn.Module):
    def __init__(self, in_channels, stride=2, M=2, r=8, L=4, use_bn = True) -> None:
        super().__init__()

        d = max(in_channels // r, L)
        self.M = M
        out_channels = in_channels
        self.out_channels = out_channels
        batchnorm = nn.BatchNorm1d if use_bn else identity
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=1, bias=False),
                batchnorm(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Conv1d(out_channels, d, 1, bias=False),
            batchnorm(d),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Conv1d(d, out_channels * M, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(inputs))
        output = torch.stack(output,dim=1)
        U = output.sum(dim=1)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)
        a_b = self.softmax(a_b)
        return torch.sum(torch.mul(a_b, output), dim=1)

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)
    def forward(self, X):
        out = X.permute(0,2,1)
        out = self.ln(out)
        out = out.permute(0,2,1)
        return out

class GlobalResponseNorm(nn.Module):
    def __init__(self, num_hiddens) -> None:
        super().__init__()
        self.num_hiddens = num_hiddens
        self.weight = nn.Parameter(torch.zeros(self.num_hiddens))
        self.bias = nn.Parameter(torch.zeros(self.num_hiddens))
    def forward(self, X):
        X_g = X.norm(p=2, dim=2, keepdim=True) # B x C x S
        X_n = X_g / (X_g.mean(dim=1, keepdim=True) + 1e-6)
        return X + torch.addcmul(self.bias.view(1,-1,1), self.weight.view(1,-1,1), X * X_n)

class mutil_scaler(nn.Module):
    def __init__(self, in_channels, out_channels, inplace=True) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1, padding='same')
        self.conv1x3 = nn.Conv1d(in_channels, out_channels, 3, padding='same')
        self.conv1x5 = nn.Conv1d(in_channels, out_channels, 5, padding='same')
        self.inplace = inplace
        self.grn = GlobalResponseNorm(in_channels)
    def forward(self, X):
        out = self.grn(X)
        out1x1 = self.conv1x1(out)                                               
        out1x3 = self.conv1x3(out)
        out1x5 = self.conv1x5(out)                                                                                                                                                                                                                                    
        #outmix = out1x1 * out1x3 * out1x5
        out = torch.cat(tensors=(out1x1, out1x3, out1x5), dim=1)
        out = F.hardswish(out, self.inplace)
        return out#

class mutil_excitation(nn.Module):
    def __init__(self, num_channels, inplace=True) -> None:
        super().__init__()    
        self.inplace = inplace
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 3, 2, 1),
            nn.Hardswish(inplace)
            ))
        self.convs.append(nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 5, 2, 2),
            nn.Hardswish(inplace)
        ))
        M = len(self.convs)
        self.gf2fs = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),# B x C x 1
            nn.Conv1d(num_channels, num_channels * M, 1, 1),
            nn.ReLU(inplace),
            nn.Unflatten(1,(M,num_channels)),
            nn.Softmax(1)
        )
    def forward(self, X):
        outs = []
        for layer in self.convs:
            outs.append(layer(X))
        outs = torch.stack(outs, dim=1)
        out = torch.sum(outs, dim=1)
        out = self.gf2fs(out)
        return torch.sum(torch.mul(out, outs), dim=1)

    
class mutil_squeeze(nn.Module):
    def __init__(self, in_channels, out_channels, inplace=True) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.inplace = inplace
        self.acti = nn.ReLU(inplace)
    def forward(self, X):
        return self.acti(self.conv1x1(X))


class mutil_block(nn.Module):
    def __init__(self, in_channels, out_channels, inplace=True, depth = 0) -> None:
        super().__init__()
        self.model = nn.Sequential(
            mutil_scaler(in_channels, in_channels),
            mutil_excitation(in_channels * 3),
            mutil_squeeze(in_channels * 3, out_channels),
        )
        self.conv1x3 = nn.Conv1d(in_channels, out_channels, 3, 2, 1)
        
        self.inplace = inplace
        self.acti = nn.ReLU(inplace)
    def forward(self, X):
        return self.model(X) + self.acti(self.conv1x3(X))
    

class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, inplace=True) -> None:
        super().__init__()
        self.name = '6mA'
        self.emb = nn.Sequential(
            nn.LazyLinear(embedding_dim, False)
        )
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            mutil_block(embedding_dim, embedding_dim * 2),
            nn.Dropout(0.2),
            mutil_block(embedding_dim * 2, embedding_dim * 4),
            nn.Dropout(0.2),
            mutil_block(embedding_dim * 4, embedding_dim * 8),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 48,embedding_dim * 48),
            nn.Hardswish(inplace),
        )
        #self.lstm = nn.LSTM(input_size = embedding_dim // 2,hidden_size = embedding_dim // 2, bidirectional=True,batch_first=True,num_layers=1)

    def forward(self, X):
        out = self.emb(X)
        out = out.permute(0,2,1)
        out = self.model(out)
        return out

class classifier2(nn.Module):
    def __init__(self, embedding_dim, inplace = True) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.LazyLinear(128),
            nn.Hardswish(inplace),
            nn.Dropout(),
            nn.Linear(128,2)
        )
    def forward(self, X):
        return self.model(X)
    