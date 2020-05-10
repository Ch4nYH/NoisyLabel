import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from losses import sample_wise_kl
from datasets import MNISTDataset, CIFARDataset, CIFAR100Dataset
from utils import get_args, accuracy, get_val_samples, WLogger, ScalarLogger
from meta_models import Model, to_var
from meta_resnet import resnet34, VNet
from pdb import set_trace as bp
from tensorboardX import SummaryWriter
from torchvision import transforms
from collections import defaultdict
import copy
import argparse
args = None

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    ##### 
    parser.add_argument('-l', '--lr', type=float, default = 1e-1 )
    parser.add_argument('-d', '--dataset', type=str, default = 'cifar10' )
    parser.add_argument('-g', '--gpu', type=str, default = "" )
    parser.add_argument('-m', '--modeldir', type=str)
    parser.add_argument('-b', '--batch-size', type=int, default = 100)
    parser.add_argument('-j', '--num-workers', type = int, default = 8 )
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-e', '--epochs', type=int, default = 100 )
    parser.add_argument('-p', '--percent', type=float, default = 0.5 )  
    parser.add_argument('-a', '--arch', type=str, default = "resnet34")
    parser.add_argument('--clamp', action="store_true" )
    parser.add_argument('--gamma', type=float, default = 1.0 )
    parser.add_argument('--prefix', default="models", type=str )
    parser.add_argument('--val-batch-size', default = None, type = int)
    parser.add_argument('--with-kl', action="store_true")
    parser.add_argument('--reg-start', type = int)

    args = parser.parse_args()
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
        
    return args
    
def main():
    global args
    args = get_args()
    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()
    print(args)
    
    if len(args.gpu) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        use_CUDA = True
    else:
        use_CUDA = False

    cudnn.benchmark = True

    data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        } 
    
    if args.dataset == 'mnist':
        train_dataset = MNISTDataset(split = 'train', seed = args.seed)
        val_dataset = MNISTDataset(split = 'val', seed = args.seed)
        num_classes = 10
        input_channel = 1
    elif args.dataset == 'cifar10':
        train_dataset = CIFARDataset(split = 'train', seed = args.seed, transform = data_transforms['train'], percent = args.percent)
        val_dataset = CIFARDataset(split = 'val', seed = args.seed, transform = data_transforms['val'], percent = args.percent)
        num_classes = 10
        input_channel = 3
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100Dataset(split = 'train', seed = args.seed, transform = data_transforms['train'], percent = args.percent)
        val_dataset = CIFAR100Dataset(split = 'val', seed = args.seed, transform = data_transforms['val'], percent = args.percent)
        num_classes = 100
        input_channel = 3
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers)

    model = get_model(args, num_classes, input_channel)
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    save_path = os.path.join(args.prefix, args.modeldir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    writer = SummaryWriter(save_path)
    
    best_prec = 0
    for epoch in range(args.epochs):
        train(model, input_channel, optimizer, criterion, train_loader, val_loader, epoch, writer, args, use_CUDA = use_CUDA, clamp = args.clamp, num_classes = num_classes)
        loss, prec = val(model, val_loader, criterion, epoch, writer, use_CUDA)
        torch.save(model, os.path.join(save_path, 'checkpoint.pth.tar'))
        if prec > best_prec:
            torch.save(model, os.path.join(save_path, 'model_best.pth.tar'))
            best_prec = prec
        
        #adjust_learning_rate(optimizers, args.lr, args.gamma, epoch, True)


def train(model, input_channel, optimizer, criterion, train_loader, val_loader, epoch, writer, args, use_CUDA = True, clamp = False, num_classes = 10):
    model.train()
    accs = []
    losses_w1 = []
    losses_w2 = []
    iter_val_loader = iter(val_loader)
    meta_criterion = nn.CrossEntropyLoss(reduce = False)
    index = 0
    noisy_labels = []
    true_labels = []

    w = defaultdict()
    w_logger = defaultdict()
    losses_logger = defaultdict()
    accuracy_logger = ScalarLogger(prefix = 'accuracy')
         
    for (input, label, real) in train_loader:
        noisy_labels.append(label)
        true_labels.append(real)
        input = to_var(input, requires_grad = False)
        label = to_var(label, requires_grad = False).long()
        index += 1
        output = model(input)
        loss = meta_criterion(output, label).sum() / input.shape[0]
        prediction = torch.softmax(output, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        top1 = accuracy(prediction, label)
        accuracy_logger.update(top1)
        
    noisy_labels = torch.cat(noisy_labels)
    true_labels = torch.cat(true_labels)
    mask = (noisy_labels != true_labels).cpu().numpy()
    for c in components:
        w_logger[c].write(writer, c, epoch)
        w_logger[c].mask_write(writer, c, epoch, mask)
        losses_logger[c].write(writer, c, epoch)

    accuracy_logger.write(writer, 'train', epoch)
    
    print("Training Epoch: {}, Accuracy: {}".format(epoch, accuracy_logger.avg()))
    return accuracy_logger.avg()

def val(model, val_loader, criterion, epoch, writer, use_CUDA = True):
    model.eval()
    accuracy_logger = ScalarLogger(prefix = 'accuracy')
    losses_logger = ScalarLogger(prefix = 'loss')
    with torch.no_grad():
        for (input, label, _) in val_loader:
            input = to_var(input, requires_grad = False)
            label = to_var(label, requires_grad = False).long()

            output = model(input)
            loss = criterion(output, label)
            prediction = torch.softmax(output, 1)
            top1 = accuracy(prediction, label)
            accuracy_logger.update(top1)
            losses_logger.update(loss)

    accuracy_logger.write(writer, 'val', epoch)
    losses_logger.write(writer, 'val', epoch)
    accuracy_ = accuracy_logger.avg()
    losses = losses_logger.avg()
    print("Validation Epoch: {}, Accuracy: {}, Losses: {}".format(epoch, accuracy_, losses))
    return accuracy_, losses

def get_model(args, num_classes, input_channel):
    if args.arch == 'default':
        return Model(num_classes, input_channel)
    elif args.arch == 'resnet34':
        return resnet34(num_classes = num_classes)
    else: 
        raise NotImplementedError

def get_optimizers(model, components, lr, gamma):
    optimizers = defaultdict()
    opt = torch.optim.Adam
    if 'all' in components:
        optimizers['all'] = opt(model.parameters(), lr = lr)
    if 'fc' in components:
        optimizers['fc'] = opt(model.fc.parameters(), lr = lr * gamma)
    if 'backbone' in components:
        optimizers['backbone'] = opt(model.backbone.parameters(), lr = lr * gamma)
    return optimizers

def adjust_learning_rate(optimizers, lr, gamma, epoch, dynamic_gamma = False):
    gamma_ = gamma / (epoch + 1) if dynamic_gamma else gamma
    for c in optimizers.keys():
        if c == 'all':
            pass # TODO: decay
        else:
            for param_group in optimizers[c].param_groups:
                param_group['lr'] = lr * gamma_
            
                
if __name__ == '__main__':
    main()
