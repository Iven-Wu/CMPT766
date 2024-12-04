
import numpy as np
import torch


def LBS(x, W1, T, R):
    bx = (T @ x.T).permute(0, 3, 1, 2)
    wbx = W1[None, :, :, None] * bx
    wbx = wbx.permute(0, 2, 1, 3)
    wbx = wbx.sum(1, keepdim=True)
    wbx = (R @ (wbx[:, 0].permute(0, 2, 1))).permute(0, 2, 1)

    return wbx

def LBS_notrans(x, W1, T):
    final_wbx = torch.zeros_like(x, requires_grad=True, device="cuda")
    sum_weight = torch.zeros_like(W1[None, :, [0]])
    for b_ind in range(T.shape[1]):
        final_wbx = final_wbx + T[:, [b_ind]].act(x) * W1[None, :, [b_ind]]
        sum_weight += W1[None, :, [b_ind]]
    return final_wbx

UNKNOWN_FLOW_THRESH = 1e7

def distance_matrix(centers):
    X = centers.T
    m, n = X.shape
    G = np.dot(X.T, X)
    D = np.zeros([n, n])

    for i in range(n):
        D[i, i] = 100
        for j in range(i+1, n):
            D[i,j] = G[i,i] - 2 * G[i,j] + G[j,j]
            D[j,i] = D[i,j]
    return D
