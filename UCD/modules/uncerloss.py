# -*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn.functional as F
from modules import transform, resnet, network, contrastive_loss,my_evidence_loss,my_sdm_loss,computeLabel
from torch.nn.functional import normalize
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1";


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(num_classes, alpha,batch_size):
    beta = torch.ones([1, num_classes], dtype=torch.float32).to(alpha.device)  # self.num_classes：类别数
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
          torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    # print("alpha",dg1)  #alpha torch.Size([128, 128])
    # print("beta",dg0)   #beta torch.Size([1, 10])
    beta = torch.ones([batch_size, batch_size], dtype=torch.float32).to(alpha.device)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    kl = kl.cuda()
    return kl



def loglikelihood_loss(y, alpha):
    y = y.cuda()
    alpha = alpha.cuda()
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    loglikelihood_var = loglikelihood_var.cuda()
    loglikelihood_err = loglikelihood_err.cuda()

    return loglikelihood_err, loglikelihood_var


def mse_loss(y, alpha, annealing_coef, num_classes, batch_size, with_kldiv=1, with_avuloss=0):   #loss_type='mse'
    """Used only for loss_type == 'mse'
    y: the one-hot labels (batchsize, num_classes)
    alpha: the predictions (batchsize, num_classes)
    epoch_num: the current training epoch
    """
    y = y.cuda()
    alpha = alpha.cuda()

    losses = {}
    loglikelihood_err, loglikelihood_var = loglikelihood_loss(y, alpha)

    losses.update({'loss_cls': loglikelihood_err, 'loss_var': loglikelihood_var})

    losses.update({'lambda': annealing_coef})      #True
    if with_kldiv==1:
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * \
                 kl_divergence(num_classes,kl_alpha,batch_size)
        losses.update({'loss_kl': kl_div})

    if with_avuloss==1:       #False
        S = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength
        pred_score = alpha / S
        uncertainty = num_classes / S
        # avu_loss = annealing_coef *
    return losses


def ce_loss(target, y, alpha, annealing_coef, num_classes,batch_size):
    """Used only for loss_type == 'ce'
    target: c (batchsize,)
    alpha: the predictions (batchsize, num_classes), alpha >= 1
    epoch_num: the current training epoch
    """
    losses = {}
    # (1) the classification loss term
    S = torch.sum(alpha, dim=1, keepdim=True)
    pred_score = alpha / S
    loss_cls = F.nll_loss(torch.log(pred_score), target, reduction='none')
    losses.update({'loss_cls': loss_cls})

    # (2) the likelihood variance term  似然方差
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    losses.update({'loss_var': loglikelihood_var})

    # (3) the KL divergence term
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
             kl_divergence(num_classes,kl_alpha,batch_size)
    losses.update({'loss_kl': kl_div, 'lambda': annealing_coef})
    return losses


def edl_loss(func, y, alpha, annealing_coef, target, num_classes, batch_size, eps = 1e-10, with_kldiv=0, with_avuloss=1):
    """Used for both loss_type == 'log' and loss_type == 'digamma'
    func: function handler (torch.log, or torch.digamma)
    y: the one-hot labels (batch_size, num_classes)
    alpha: the predictions (batch_size, num_classes)
    epoch_num: the current training epoch
    """
    alpha = alpha.cuda()
    y = y.cuda()
    target = target.cuda()
    # num_classes = num_classes.cuda()

    losses = {}
    S = torch.sum(alpha, dim=1, keepdim=True)
    S = S.cuda()
    uncertainty = num_classes / S

    label_num = torch.sum(y, dim=1, keepdim=True)

    temp = (1 / alpha) * y
    g = (1 - uncertainty.detach()) * label_num * torch.div(temp, torch.sum(temp, dim=1, keepdim=True)) #对应3.3节中，使用u生成软标签的公式.  div做除法
    A = torch.sum(g * (func(S) - func(alpha)), dim=1, keepdim=True)

    losses.update({'loss_cls': A})
    losses.update({'lambda': annealing_coef})
    if with_kldiv==1:
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * \
                 kl_divergence(num_classes,kl_alpha,batch_size)
        losses.update({'loss_kl': kl_div})

    if with_avuloss==1:
        pred = alpha / S
        uncertainty = num_classes / S
        inacc_measure = torch.abs(pred - target).sum(dim=1) / 2.0

        acc_uncertain = - (torch.ones_like(inacc_measure) - inacc_measure) * torch.log(1 - uncertainty + eps)
        inacc_certain = - inacc_measure * torch.log(uncertainty + eps)

        batch_size, _ = y.shape
        inacc_measure_bool = inacc_measure.clone()
        inacc_measure_bool[inacc_measure_bool > 0.7] = 1
        inacc_measure_bool[inacc_measure_bool <= 0.7] = 0
        acc_match = 1 - torch.sum(inacc_measure_bool) / batch_size

        avu_loss = annealing_coef * acc_match * acc_uncertain + (1 - annealing_coef) * (
                1 - acc_match) * inacc_certain

        losses.update({'loss_avu': avu_loss})
    return losses


def compute_annealing_coef(self, **kwargs):
    assert 'epoch' in kwargs, "epoch number is missing!"
    assert 'total_epoch' in kwargs, "total epoch number is missing!"
    epoch_num, total_epoch = kwargs['epoch'], kwargs['total_epoch']
    # annealing coefficient
    if self.annealing_method == 'step':
        annealing_coef = torch.min(torch.tensor(
            1.0, dtype=torch.float32), torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32))
    elif self.annealing_method == 'exp':
        annealing_start = torch.tensor(self.annealing_start, dtype=torch.float32)
        annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * epoch_num)
    else:
        raise NotImplementedError
    annealing_coef = annealing_coef.cuda()
    return annealing_coef

def my_compute_annealing_coef(annealing_method,epoch_num, total_epoch,annealing_step=10,annealing_start=0.01):
    # annealing coefficient
    if annealing_method == 'step':
        annealing_coef = torch.min(torch.tensor(
            1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))
    elif annealing_method == 'exp':
        annealing_start = torch.tensor(annealing_start, dtype=torch.float32)
        annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * epoch_num)
    else:
        raise NotImplementedError
    annealing_coef = annealing_coef.cuda()
    return annealing_coef



def my_SUM_loss(output,target,epoch_num,max_epochs,evidence,loss_type,num_classes,batch_size):  
    if evidence == 'relu':
        evidence = relu_evidence(output).cuda()
    elif evidence == 'exp':
        evidence = exp_evidence(output)
    elif evidence == 'softplus':
        evidence = softplus_evidence(output)
    else:
        raise NotImplementedError
    alpha = evidence + 1
    y = target
    annealing_coef = my_compute_annealing_coef('exp',epoch_num,max_epochs)
    if loss_type == 'mse':
        results =  mse_loss(y, alpha, annealing_coef,num_classes,batch_size)
    elif  loss_type == 'log':
        results =  edl_loss(torch.log, y, alpha, annealing_coef, target, num_classes, batch_size)
    elif  loss_type == 'digamma':
        results =  edl_loss(torch.digamma, y, alpha, annealing_coef, target, num_classes, batch_size)
    elif  loss_type == 'cross_entropy':
        results =  ce_loss(target, y, alpha, annealing_coef)
    else:
        raise NotImplementedError

    # uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    # results.update({'uncertainty': uncertainty})
    # return results

    # results.get('loss_kl') +
    if loss_type == 'log':
        loss =(torch.sum((torch.sum(results.get('loss_cls') + results.get('loss_avu'), dim=0, keepdim=True)/batch_size),dim=1)/batch_size).cuda() #因为选择edl_Loss时，得到的loss是个矩阵，所以要两次sum
    elif loss_type == 'mse':
        loss = (torch.sum(results.get('loss_cls') + results.get('loss_kl'), dim=0, keepdim=True) / batch_size).cuda()

    return loss


