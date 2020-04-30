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
from meta_resnet import resnet34
from pdb import set_trace as bp
from tensorboardX import SummaryWriter
from torchvision import transforms
from collections import defaultdict
import copy
args = None
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
        input_channel = 1
    elif args.dataset == 'cifar10':
        train_dataset = CIFARDataset(split = 'train', seed = args.seed, transform = data_transforms['train'], percent = args.percent)
        val_dataset = CIFARDataset(split = 'val', seed = args.seed, transform = data_transforms['val'], percent = args.percent)
        input_channel = 3
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100Dataset(split = 'train', seed = args.seed, transform = data_transforms['train'], percent = args.percent)
        val_dataset = CIFAR100Dataset(split = 'val', seed = args.seed, transform = data_transforms['val'], percent = args.percent)
        input_channel = 3
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers)

    model = get_model(args, input_channel = input_channel, num_classes=args.num_classes)
    
    optimizers = get_optimizers(model, args.components, args.lr, args.gamma)
       
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80], gamma=0.5, last_epoch=-1)

    save_path = os.path.join(args.prefix, args.modeldir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    writer = SummaryWriter(save_path)

    best_prec = 0
    for epoch in range(args.epochs):
        train(model, input_channel, optimizers, criterion, args.components, train_loader, val_loader, epoch, writer, args, use_CUDA = use_CUDA, clamp = args.clamp, num_classes = args.num_classes)
        loss, prec = val(model, val_loader, criterion, epoch, writer, use_CUDA)
        torch.save(model, os.path.join(save_path, 'checkpoint.pth.tar'))
        if prec > best_prec:
            torch.save(model, os.path.join(save_path, 'model_best.pth.tar'))
            best_prec = prec
        
        adjust_learning_rate(optimizers, args.lr, args.gamma, epoch, True)


def train(model, input_channel, optimizers, criterion, components, train_loader, val_loader, epoch, writer, args, use_CUDA = True, clamp = False, num_classes = 10):
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
    for c in components:
        w[c] = None
        w_logger[c] = WLogger()
        losses_logger[c] = ScalarLogger(prefix = 'loss')
         
    for (input, label, real) in train_loader:
        noisy_labels.append(label)
        true_labels.append(real)
        
        meta_model = get_model(args, num_classes = num_classes, input_channel = input_channel)
        meta_model.load_state_dict(model.state_dict())
        if use_CUDA:
            meta_model = meta_model.cuda()

        
        val_input, val_label, iter_val_loader = get_val_samples(iter_val_loader, val_loader)
        input = to_var(input, requires_grad = False)
        label = to_var(label, requires_grad = False).long()
        val_input = to_var(val_input, requires_grad = False)
        val_label = to_var(val_label, requires_grad = False).long()
        
        meta_output = meta_model(input)
        cost = meta_criterion(meta_output, label) 
        eps = to_var(torch.zeros(cost.size()))
        meta_loss = (cost * eps).sum()
        meta_model.zero_grad()
        
        if 'all' in components:
            grads = torch.autograd.grad(meta_loss, (meta_model.parameters()), create_graph=True)
            meta_model.update_params(0.001, source_params = grads)     

            meta_val_output = meta_model(val_input)
            meta_val_loss = meta_criterion(meta_val_output, val_label).sum()
            grad_eps = torch.autograd.grad(meta_val_loss, eps, only_inputs = True)[0]
            if clamp:
                w['all'] = torch.clamp(-grad_eps, min = 0)
            else:
                w['all'] = -grad_eps
            
            norm = torch.sum(abs(w['all']))
            assert (clamp and len(components) == 1) or (len(components) > 1), "Error combination"
            w['all'] = w['all'] / norm
            if ('fc' in components):
                w['fc'] = copy.deepcopy(w['all'])
                w['fc'] = torch.clamp(w['fc'], max = 0)
                w['all'] = torch.clamp(w['all'], min = 0)
            elif ('backbone' in components):
                w['backbone'] = copy.deepcopy(w['all'])
                w['backbone'] = torch.clamp(w['backbone'], max = 0)
                w['all'] = torch.clamp(w['all'], min = 0)
            
        else:
            assert ('backbone' in components) and ('fc' in components)
            
            grads_backbone = torch.autograd.grad(meta_loss, (meta_model.backbone.parameters()), create_graph=True, retain_graph = True)
            grads_fc = torch.autograd.grad(meta_loss, (meta_model.fc.parameters()), create_graph=True)
            
            # Backbone Grads
            meta_model.backbone.update_params(0.001, source_params = grads_backbone)
            meta_val_feature = torch.flatten(meta_model.backbone(val_input), 1)
            meta_val_output = meta_model.fc(val_input)
            meta_val_loss = meta_criterion(meta_val_output, val_label).sum()
            
            if args.with_kl:
                train_feature = torch.flatten(meta_model.backbone(input), 1)
                meta_val_loss -= sample_wise_kl(train_feature, meta_val_feature)
            grad_eps = torch.autograd.grad(meta_val_loss, eps, only_inputs = True, retain_graph = True)[0]
            if clamp:
                w['backbone'] = torch.clamp(-grad_eps, min = 0)
            else:
                w['backbone'] = -grad_eps
            norm = torch.sum(abs(w['backbone']))
            w['backbone'] = w['backbone'] / norm

            # FC backward
            meta_model.load_state_dict(model.state_dict())
            meta_model.fc.update_params(0.001, source_params = grads_fc)
            meta_val_output = meta_model(val_input)
            meta_val_loss = meta_criterion(meta_val_output, val_label).sum()
            grad_eps = torch.autograd.grad(meta_val_loss, eps, only_inputs = True, retain_graph = True)[0]
            
            if clamp:
                w['fc'] = torch.clamp(-grad_eps, min = 0)
            else:
                w['fc'] = -grad_eps
            norm = torch.sum(abs(w['fc']))
            w['fc'] = w['fc'] / norm
                
            
        index += 1
        output = model(input)
        loss = defaultdict()
        prediction = torch.softmax(output, 1)
        for c in components:
            w_logger[c].update(w[c])
            loss[c] = (meta_criterion(output, label) * w[c]).sum()
            optimizers[c].zero_grad()
            loss[c].backward(retain_graph = True)
            optimizers[c].step()
            losses_logger[c].update(loss[c])

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

def get_model(args, num_classes = 10, input_channel = 3):
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
        optimizers['all'] = opt(model.parameters(), lr = args.lr)
    if 'fc' in components:
        optimizers['fc'] = opt(model.fc.parameters(), lr = args.lr * args.gamma)
    if 'backbone' in components:
        optimizers['backbone'] = opt(model.backbone.parameters(), lr = args.lr * args.gamma)
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
