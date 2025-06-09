"""
RealNVP 仿射耦合层
"""

import torch
import torch.nn as nn



class LayerNormFlow(nn.Module):

    def __init__(self, M, N, eps=1e-5):
        super(LayerNormFlow, self).__init__()
        self.log_gamma = nn.Parameter(torch.zeros(M+N))
        self.beta = nn.Parameter(torch.zeros(M+N))
        self.eps = eps

    def forward(self, M, N, invert=False):

        inputs = torch.cat([M, N], dim=-1)
        if not invert:

            self.batch_mean = inputs.mean(-1)

            self.batch_var = (inputs - self.batch_mean.reshape(-1, 1)).pow(2).mean(-1) + self.eps

            mean = self.batch_mean
            var = self.batch_var

            x_hat = (inputs - mean.reshape(-1, 1)) / var.sqrt().reshape(-1, 1)
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y[:, :M.shape[-1]], y[:, -N.shape[-1]:]
        else:
            mean = self.batch_mean
            var = self.batch_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt().reshape(-1, 1) + mean.reshape(-1, 1)

            return y[:, :M.shape[-1]], y[:, -N.shape[-1]:]


class CoupleLayer(nn.Module):
    def __init__(self, M, N, fn, bn, key=0, reverse=False):
        super(CoupleLayer, self).__init__()
        self.reverse = reverse

        self.S = nn.Sequential()
        self.T = nn.Sequential()
        self.key = key

        if not reverse:
            for i in range(len(fn) + 1):
                if i == 0:
                    self.S.add_module('layer_f{}'.format(i), nn.Linear(M, fn[0]))
                    self.T.add_module('layer_f{}'.format(i), nn.Linear(M, fn[0]))
                    self.S.add_module('Activation_f{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_f{}'.format(i), nn.LeakyReLU(0.5))
                elif i < len(fn):
                    self.S.add_module('layer_f{}'.format(i), nn.Linear(fn[i - 1], fn[i]))
                    self.T.add_module('layer_f{}'.format(i), nn.Linear(fn[i - 1], fn[i]))
                    self.S.add_module('Activation_f{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_f{}'.format(i), nn.LeakyReLU(0.5))
                elif i == len(fn):
                    self.S.add_module('layer_f{}'.format(i), nn.Linear(fn[i - 1], N))
                    self.T.add_module('layer_f{}'.format(i), nn.Linear(fn[i - 1], N))
                    self.T.add_module('Activation_f{}'.format(i), nn.Tanh())

        else:
            for i in range(len(bn) + 1):
                if i == 0:
                    self.S.add_module('layer_b{}'.format(i), nn.Linear(N, bn[0]))
                    self.T.add_module('layer_b{}'.format(i), nn.Linear(N, bn[0]))
                    self.S.add_module('Activation_b{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_b{}'.format(i), nn.LeakyReLU(0.5))
                elif i < len(bn):
                    self.S.add_module('layer_b{}'.format(i), nn.Linear(bn[i - 1], bn[i]))
                    self.T.add_module('layer_b{}'.format(i), nn.Linear(bn[i - 1], bn[i]))
                    self.S.add_module('Activation_b{}'.format(i), nn.Tanh())
                    self.T.add_module('Activation_b{}'.format(i), nn.LeakyReLU(0.5))
                elif i == len(bn):
                    self.S.add_module('layer_b{}'.format(i), nn.Linear(bn[i - 1], M))
                    self.T.add_module('layer_b{}'.format(i), nn.Linear(bn[i - 1], M))
                    self.T.add_module('Activation_b{}'.format(i), nn.Tanh())
        self._init_weight()

    def _init_weight(self):
        for i in range(len(self.S)):
            if i % 2 == 0:
                nn.init.xavier_uniform_(self.S[i].weight)
                nn.init.zeros_(self.S[i].bias)
        for i in range(len(self.T)):
            if i % 2 == 0:
                nn.init.xavier_uniform_(self.T[i].weight)
                nn.init.zeros_(self.T[i].bias)

    def forward(self, input_M, input_N, invert=False):
        if not invert:
            if not self.reverse:
                output_M = input_M
                if self.key == 0:
                    S_out = torch.exp(self.S(input_M))
                elif self.key == 1:
                    S_out = -torch.exp(self.S(input_M))
                T_out = self.T(input_M)
                output_N = input_N * S_out + T_out
            else:
                output_N = input_N
                if self.key == 0:
                    S_out = torch.exp(self.S(input_N))
                elif self.key == 1:
                    S_out = -torch.exp(self.S(input_N))

                T_out = self.T(input_N)
                output_M = input_M * S_out + T_out
        else:
            if self.reverse:
                output_N = input_N
                if self.key == 0:
                    S_out = torch.exp(-self.S(input_N))
                elif self.key == 1:
                    S_out = -torch.exp(-self.S(input_N))
                T_out = self.T(input_N)
                output_M = (input_M - T_out) * S_out
            else:
                output_M = input_M
                if self.key == 0:
                    S_out = torch.exp(-self.S(input_M))
                elif self.key == 1:
                    S_out = -torch.exp(-self.S(input_M))
                T_out = self.T(input_M)
                output_N = (input_N - T_out) * S_out

        return output_M, output_N


class RealNVP(nn.Module):
    def __init__(self, M, N, n_layers, fn, bn):
        super(RealNVP, self).__init__()

        self.M = M
        self.N = N
        self.spatial = 7*7
        self.M_dim = M * self.spatial
        self.N_dim = N * self.spatial
        layers = []
        for i in range(n_layers):
            layers.append(CoupleLayer(self.M_dim, self.N_dim, fn, bn, key=0, reverse=False))
            layers.append(CoupleLayer(self.M_dim, self.N_dim, fn, bn, key=0, reverse=True))
            layers.append(LayerNormFlow(self.M_dim, self.N_dim))
        self.layers = nn.ModuleList(layers)


    def forward(self, X, invert=False):
        B, C, H, W = X.shape
        flat = X.reshape(B, -1)
        xm, xn = flat[:, :self.M_dim], flat[:, self.M_dim:]
        if invert:
            for layer in reversed(self.layers):
                xm, xn = layer(xm, xn, invert=True)
        else:
            for layer in self.layers:
                xm, xn = layer(xm, xn, invert=False)
        out = torch.cat([xm, xn], dim=1).reshape(B, C, H, W)
        return out


if __name__ == '__main__':
    model = RealNVP(9, 64, 2, [64,128], [32])
    a = torch.tensor(2)
    action_onehot = torch.zeros(1, 9, 7, 7)
    action_onehot[0, a, :, :] = 1
    s = torch.randint(1, 10, (1, 64, 7, 7), dtype=torch.float32)
    i = torch.cat([action_onehot,s],dim=1)
    print(i)
    o = model(i, invert=False)
    print(o)
    print(model(o, invert=True))
