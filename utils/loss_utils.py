import torch
import torch.nn as nn
import random
import numpy as np

seed = 2000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class ARAPLoss(nn.Module):
    def __init__(self, points, faces, average=False):
        super(ARAPLoss, self).__init__()
        self.nv = points.shape[0]
        self.nf = faces.shape[0]
        # faces -= 1
        self.average = average
        laplacian = np.zeros([self.nv, self.nv], dtype=np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, dx, x):
        # lap: Nv Nv
        # dx: N, Nv, 3
        diffx = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device="cuda")
        diffdx = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device="cuda")
        for i in range(3):
            dx_sub = self.laplacian.matmul(torch.diag_embed(dx[:, :, i]))  # N, Nv, Nv)
            dx_diff = (dx_sub - dx[:, :, i:i + 1])

            x_sub = self.laplacian.matmul(torch.diag_embed(x[:, :, i]))  # N, Nv, Nv)
            x_diff = (x_sub - x[:, :, i:i + 1])

            diffdx += (dx_diff).pow(2)
            diffx += (x_diff).pow(2)

        diff = (diffx - diffdx).abs()
        diff = torch.stack([diff[i][self.laplacian.bool()].mean() for i in range(x.shape[0])])
        return diff


class LaplacianLoss(nn.Module):
    def __init__(self, points, faces, average=False):
        super().__init__()
        self.nv = points.shape[0]
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv], dtype=np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, x):
        # lap: Nv Nv
        # dx: N, Nv, 3

        x_sub = (self.laplacian @ x / (self.laplacian.sum(0)[None, :, None] + 1e-6))
        x_diff = (x_sub - x)
        x_diff = (x_diff).pow(2)
        return torch.mean(x_diff)


class Preframe_ARAPLoss(nn.Module):
    def __init__(self, points, faces, average=False):
        super().__init__()
        self.nv = points.shape[0]
        self.nf = faces.shape[0]
        # faces -= 1
        self.average = average
        laplacian = np.zeros([self.nv, self.nv], dtype=np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, dx, x):
        # lap: Nv Nv
        # dx: N, Nv, 3
        diffx = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device="cuda")
        diffdx = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device="cuda")
        for i in range(3):
            dx_sub = self.laplacian.matmul(torch.diag_embed(dx[:, :, i]))  # N, Nv, Nv)
            dx_diff = (dx_sub - dx[:, :, i:i + 1])

            x_sub = self.laplacian.matmul(torch.diag_embed(x[:, :, i]))  # N, Nv, Nv)
            x_diff = (x_sub - x[:, :, i:i + 1])

            diffdx += (dx_diff).pow(2)
            diffx += (x_diff).pow(2)

        diff = (diffx - diffdx).abs()
        diff = (diff * (self.laplacian.bool()[None])).sum(2)

        return diff


class Preframe_LaplacianLoss(nn.Module):
    def __init__(self, points, faces, average=False):
        super().__init__()
        self.nv = points.shape[0]
        self.nf = faces.shape[0]
        # faces -= 1
        self.average = average
        laplacian = np.zeros([self.nv, self.nv], dtype=np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, x):
        # lap: Nv Nv
        # dx: N, Nv, 3

        x_sub = (self.laplacian @ x / (self.laplacian.sum(0)[None, :, None] + 1e-6))
        x_diff = (x_sub - x)
        x_diff = (x_diff).pow(2)

        return x_diff.mean(2)


class OffsetNet(nn.Module):
    def __init__(self, input_ch=3, out_ch=3, W=256):
        super(OffsetNet, self).__init__()
        self.W = W
        self.input_ch = input_ch
        self.out_ch = out_ch

        self.layers = nn.Sequential(
            nn.Linear(input_ch, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, out_ch)
        )

    def forward(self, x):
        return self.layers(x)
