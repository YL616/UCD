import os
import numpy as np
import torch
import torchvision
import argparse
from collections import OrderedDict

from modules import transform, resnet, network,uncerloss, corres, Uncertainty
from utils import yaml_config_hook
from torch.utils import data
import torch.utils.data.distributed
from evaluation import evaluation
from train import train_net



os.environ["CUDA_VISIBLE_DEVICES"] = "1";

def train(c_instance,c_semantic):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = c_instance(z_i, z_j)
        loss_semantic = c_semantic(c_i, c_j)
        evidence = Uncertainty.uncerevi(c_i, c_j)
        sum = uncerloss.my_SUM_loss(z_i, step, max_epochs, evidence, num_classes, batch_size)
        loss = loss_instance + loss_semantic
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


def main():
    parser = argparse.ArgumentParser()
    config = yaml_config_hook.yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
        args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


#     train_dataset = torchvision.datasets.CIFAR10(
#             root=args.dataset_dir,
#             download=True,
#             train=True,
#             transform=transform.Transforms(size=args.image_size, s=0.5),
#         )
#
#     test_dataset = torchvision.datasets.CIFAR10(
#             root=args.dataset_dir,
#             download=True,
#             train=False,
#             transform=transform.Transforms(size=args.image_size, s=0.5),
#         )
#     dataset = data.ConcatDataset([train_dataset, test_dataset])
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
#                                               pin_memory=True)
#
#     test_dataset_1 = torchvision.datasets.CIFAR10(
#         root=args.dataset_dir,
#         download=True,
#         train=True,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     test_dataset_2 = torchvision.datasets.CIFAR10(
#         root=args.dataset_dir,
#         download=True,
#         train=False,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     dataset_test = data.ConcatDataset([test_dataset_1, test_dataset_2])
#     test_loader = torch.utils.data.DataLoader(
#         dataset=dataset_test,
#         batch_size=args.test_batch_size,
#         shuffle=False)


# CIFAR-100
#     train_dataset = torchvision.datasets.CIFAR100(
#             root=args.dataset_dir,
#             download=True,
#             train=True,
#             transform=transform.Transforms(size=args.image_size, s=0.5),
#         )
#     test_dataset = torchvision.datasets.CIFAR100(
#             root=args.dataset_dir,
#             download=True,
#             train=False,
#             transform=transform.Transforms(size=args.image_size, s=0.5),
#         )
#     dataset = data.ConcatDataset([train_dataset, test_dataset])
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
#                                               pin_memory=True)
#
#     test_dataset_1 = torchvision.datasets.CIFAR100(
#         root=args.dataset_dir,
#         download=True,
#         train=True,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     test_dataset_2 = torchvision.datasets.CIFAR100(
#         root=args.dataset_dir,
#         download=True,
#         train=False,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     dataset_test = data.ConcatDataset([test_dataset_1, test_dataset_2])
#     test_loader = torch.utils.data.DataLoader(
#         dataset=dataset_test,
#         batch_size=args.test_batch_size,
#         shuffle=False)


#、ImgNet-10
#     train_dataset = torchvision.datasets.ImageFolder(
#         root='./imagenet-10',
#         transform=transform.Transforms(size=args.image_size, blur=True),
#     )
#     data_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=True)
#
#     dataset_test = torchvision.datasets.ImageFolder(
#         root='./imagenet-10',
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset=dataset_test,
#         batch_size=args.test_batch_size,
#         shuffle=False,
#         drop_last=True)

#
#     train_dataset = torchvision.datasets.STL10(
#         root=args.dataset_dir,
#         split="train",
#         download=True,
#         transform=transform.Transforms(size=args.image_size),
#     )
#     test_dataset = torchvision.datasets.STL10(
#         root=args.dataset_dir,
#         split="test",
#         download=True,
#         transform=transform.Transforms(size=args.image_size),
#     )
#     cluster_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
#     data_loader = torch.utils.data.DataLoader(
#         cluster_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         drop_last=True,
#         num_workers=args.workers,
#     )
# #
#     train_dataset1 = torchvision.datasets.STL10(
#         root=args.dataset_dir,
#         split="train",
#         download=True,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     test_dataset2 = torchvision.datasets.STL10(
#         root=args.dataset_dir,
#         split="test",
#         download=True,
#         transform=transform.Transforms(size=args.image_size).test_transform,
#     )
#     dataset = torch.utils.data.ConcatDataset([train_dataset1, test_dataset2])
#     test_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=500,
#         shuffle=False,
#         drop_last=False,
#         num_workers=args.workers,
#     )

# ImageNet_Dogs数据集
    dataset = torchvision.datasets.ImageFolder(
        root='dataset/imagenet-dogs',
        transform=transform.Transforms(size=args.image_size, blur=True),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    dataset1 = torchvision.datasets.ImageFolder(
        root='dataset/imagenet-dogs',
        transform=transform.Transforms(size=args.image_size).test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset1,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )



# -------------------------------------------------------------------------------------------------------
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    c_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)

    c_semantic = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(c_instance,c_semantic)
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)





if __name__ == "__main__":
    main()

