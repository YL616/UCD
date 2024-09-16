# -*-coding:utf-8-*-
import torch
from modules import transform, resnet, network, contrastive_loss,my_evidence_loss,my_sdm_loss,computeLabel,DELU_loss
from torch.nn.functional import normalize

def uncerevi(z_i, z_j, batch_size,num_classs,func,epoch, max_epochs=20, tau=0.1):
    cosine_similarity_matrix = torch.matmul(z_i, z_j.T)
    evidences = torch.exp(torch.tanh(cosine_similarity_matrix) / tau)
    sum_e = evidences + evidences.t()
    alpha_i2t = evidences + 1
    alpha_t2i = evidences.t() + 1
    L1 = torch.sum(alpha_i2t, dim=1, keepdim=True)
    u1 = num_classs/L1
    L2 = torch.sum(alpha_t2i, dim=1, keepdim=True)
    u2 = num_classs/L2
    uncertainty = 0.5 * u1 + 0.5 * u2
    label = computeLabel.deleteHigh_Uncertainty(cosine_similarity_matrix, uncertainty, batch_size, 12, 20, 10)
    label = label.cuda()
    S = torch.sum(alpha_i2t, dim=1, keepdim=True)
    E = alpha_i2t - 1
    m = alpha_i2t / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha_i2t * (S - alpha_i2t) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    C = alpha_i2t * my_evidence_loss.my_KL(alp, batch_size)
    D = A+B+C
    # print(D.is_leaf)
    return (A + B) + C




