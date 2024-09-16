import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import torch.nn as nn

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment

    ind = linear_sum_assignment(w.max() - w)
    accuracy = 0.0
    for i in ind[0]:
        accuracy = accuracy + w[i, ind[1][i]]
    return accuracy / y_pred.size


def get_predict_label(net,test_loader,dataset_size,batch_size = 500):
    pred_label_c = torch.zeros([dataset_size]).cuda()
    true_label_new = torch.zeros([dataset_size])
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    my_counter = 0
    feature = torch.zeros(19500, 512)
    for step, (x, y) in enumerate(test_loader):
        x = torch.tensor([item.cpu().detach().numpy() for item in x])
        x = x.cuda()
        h = net.resnet(x)  #torch.Size([128, 512])
        feature[step * 500:(step + 1) * 500, ] = h
        c = net.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        pred_label_c[my_counter * batch_size:(my_counter + 1) * batch_size] = c
        true_label_new[my_counter * batch_size:(my_counter + 1) * batch_size] = y
        my_counter += 1
    pred_label= pred_label_c.cpu().numpy()
    true_label = true_label_new.cpu().numpy()
    return feature,pred_label,true_label



def net_evaluation(net,test_loader,dataset_size,test_batch_size = 500):

    pred_label_c = torch.zeros([dataset_size]).cuda()
    true_label_new = torch.zeros([dataset_size])

    net.eval()  #evaluation
    for param in net.parameters():
        param.requires_grad = False


    my_counter = 0
    for step, (x, y) in enumerate(test_loader):
   
        x = x.cuda()
        h = net.resnet(x)

        c = net.cluster_projector(h);    
        c = torch.argmax(c, dim=1)

        pred_label_c[my_counter * test_batch_size:(my_counter + 1) * test_batch_size] = c
        true_label_new[my_counter * test_batch_size:(my_counter + 1) * test_batch_size] = y
        my_counter += 1

    my_acc = acc(true_label_new.cpu().numpy(), pred_label_c.cpu().numpy())
    my_nmi = nmi(true_label_new.cpu().numpy(), pred_label_c.cpu().numpy())
    my_ari = ari(true_label_new.cpu().numpy(), pred_label_c.cpu().numpy())

    print("ACC:", my_acc)
    print("NMI:", my_nmi)
    print("ARI:", my_ari)