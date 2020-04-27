import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets import MNISTDataset, CIFARDataset, CIFAR100Dataset
from utils import get_args, accuracy, get_val_samples
from meta_models import Model, to_var
from meta_resnet import resnet34
from pdb import set_trace as bp
from tensorboardX import SummaryWriter
from torchvision import transforms
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
    
    optimizers = get_optimizers(args.components)
    
        
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80], gamma=0.5, last_epoch=-1)

    save_path = os.path.join(args.prefix, args.modeldir)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    writer = SummaryWriter(save_path)

    best_prec = 0
    for epoch in range(args.epochs):
        train(model, input_channel, optimizers, criterion, args.components, train_loader, val_loader, epoch, writer, use_CUDA, clamp = args.clamp, num_classes = args.num_classes)
        loss, prec = val(model, val_loader, criterion, epoch, writer, use_CUDA)
        torch.save(model, os.path.join(save_path, 'checkpoint.pth.tar'))
        if prec > best_prec:
            torch.save(model, os.path.join(save_path, 'model_best.pth.tar'))
            best_prec = prec


def train(model, input_channel, optimizers, criterion, components, train_loader, val_loader, epoch, writer, use_CUDA = True, clamp = False, num_classes = 10):
    model.train()
    accs = []
    losses_w1 = []
    losses_w2 = []
    iter_val_loader = iter(val_loader)
    meta_criterion = nn.CrossEntropyLoss(reduce = False)
    index = 0
    w2 = None

    w1_all = []
    w2_all = []
    noisy_labels = []
    true_labels = []
    for (input, label, real) in train_loader:
        noisy_labels.append(label)
        true_labels.append(real)
        
        meta_model = get_model(num_classes = num_classes)
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
                w1 = torch.clamp(-grad_eps, min = 0)
            else:
                w1 = -grad_eps
            
            norm = torch.sum(abs(w1))
            w1 = w1 / norm
            if ('fc' in components) or ('backbone' in components):
                w2 = copy.deepcopy(w1)
                w2 = torch.clamp(w2, max = 0)
                w1 = torch.clamp(w1, min = 0)

            w1_all.append(w1.detach().cpu().view(-1).numpy())
            if w2 is not None:
                w2_all.append(w2.detach().cpu().view(-1).numpy())

            assert np.all((w1 >= 0).cpu().numpy())
            if w2 is not None: assert np.all((w2 <= 0).cpu().numpy())
        
        else:
            grads_backbone = torch.autograd.grad(meta_loss, (meta_model.backbone.parameters()), create_graph=True, retain_graph = True)
            grads_fc = torch.autograd.grad(meta_loss, (meta_model.fc.parameters()), create_graph=True)
            meta_model.backbone.update_params(0.001, source_params = grads_backbone)
            
            meta_val_output = meta_model(val_input)
            meta_val_loss = meta_criterion(meta_val_output, val_label).sum()
            grad_eps = torch.autograd.grad(meta_val_loss, eps, only_inputs = True, retain_graph = True)[0]
            if clamp:
                w1 = torch.clamp(-grad_eps, min = 0)
            else:
                w1 = -grad_eps
            norm = torch.sum(abs(w1))
            w1 = w1 / norm

            # FC backward
            meta_model.load_state_dict(model.state_dict())
            meta_model.classifier.update_params(0.001, source_params = grads_fc)

            meta_val_output = meta_model(val_input)
            meta_val_loss = meta_criterion(meta_val_output, val_label).sum()
            grad_eps = torch.autograd.grad(meta_val_loss, eps, only_inputs = True, retain_graph = True)[0]
            
            if clamp:
                w2 = torch.clamp(-grad_eps, min = 0)
            else:
                w2 = -grad_eps
            norm = torch.sum(abs(w2))

            w2 = w2 / norm
            w1_all.append(w1.detach().cpu().view(-1).numpy())
            if w2 is not None:
                w2_all.append(w2.detach().cpu().view(-1).numpy())
                
        index += 1
        output = model(input)
        loss_w1 = (meta_criterion(output, label) * w1).sum()
        
        prediction = torch.softmax(output, 1)

        optimizers[0].zero_grad()
        loss_w1.backward(retain_graph = True)
        optimizers[0].step()
        
        if w2 is not None:
            loss_w2 = (meta_criterion(output, label) * w2).sum()
            optimizers[1].zero_grad()
            loss_w2.backward()
            optimizers[1].step()
            losses_w2.append(loss_w2.detach())

        top1 = accuracy(prediction, label)
        accs.append(top1)
        losses.append(loss.detach())
        

    acc = sum(accs) / len(accs)
    loss = sum(losses) / len(losses)
    if len(losses_2) > 0: loss_2 = sum(losses_2) / len(losses_2)
    else: loss_2 = 0

    w1_all = np.concatenate(w1_all)
    if len(w2_all) > 0: w2_all = np.concatenate(w2_all)
    noisy_labels = torch.cat(noisy_labels)
    true_labels = torch.cat(true_labels)
    writer.add_histogram("train/w1", w1_all, epoch)
    if len(w2_all) > 0: writer.add_histogram("train/w2", w2_all, epoch)
    
    print(np.sum(w1_all[noisy_labels != true_labels] != 0))
    raise NotADirectoryError

    writer.add_scalar("train/w1_on_noisy", np.sum(w1_all[noisy_labels != true_labels] != 0), epoch)
    writer.add_scalar("train/w1_on_clean", np.sum(w1_all[noisy_labels == true_labels] != 0), epoch)
    if len(w2_all) > 0:
        writer.add_scalar("train/w2_on_noisy", np.sum(w2_all[noisy_labels != true_labels] != 0), epoch)
        writer.add_scalar("train/w2_on_clean", np.sum(w2_all[noisy_labels == true_labels] != 0), epoch)

    writer.add_histogram("train/w1_on_noisy", w1_all[noisy_labels != true_labels], epoch)
    writer.add_histogram("train/w1_on_clean", w1_all[noisy_labels == true_labels], epoch)
    if len(w2_all) > 0:
        writer.add_histogram("train/w2_on_noisy", w2_all[noisy_labels != true_labels], epoch)
        writer.add_histogram("train/w2_on_clean", w2_all[noisy_labels == true_labels], epoch)

    writer.add_scalar("train/acc", acc, epoch)
    writer.add_scalar("train/loss_w1", loss, epoch)
    if w2 is not None:
        writer.add_scalar("train/loss_w2", loss_2, epoch)
    
    print("Training Epoch: {}, Accuracy: {}, Losses: {}".format(epoch, acc, loss))
    return acc, loss

def val(model, val_loader, criterion, epoch, writer, use_CUDA = True):
    model.eval()
    accs = []
    losses = []
    with torch.no_grad():
        for (input, label, _) in val_loader:
            input = to_var(input, requires_grad = False)
            label = to_var(label, requires_grad = False).long()

            output = model(input)
            loss = criterion(output, label)
            prediction = torch.softmax(output, 1)
            top1 = accuracy(prediction, label)
            accs.append(top1)
            losses.append(loss.detach())

    acc = sum(accs) / len(accs)
    loss = sum(losses) / len(losses)
    writer.add_scalar("val/acc", acc, epoch)
    writer.add_scalar("val/loss", loss, epoch)
    print("Validation Epoch: {}, Accuracy: {}, Losses: {}".format(epoch, acc, loss))
    return acc, loss

def get_model(args, num_classes = 10, input_channel = 3):
    if args.arch == 'default':
        return Model(num_classes, input_channel)
    elif args.arch == 'resnet34':
        return resnet34(num_classes = num_classes)
    else 
        raise NotImplementedError

def get_optimizers(components, lr, gamma):
    optimizers = []
    opt = torch.optim.Adam
    if 'all' in components:
        optimizers.append(opt(model.parameters(), lr = args.lr))
    elif 'fc' in components:
        optimizers.append(opt(model.fc.parameters(), lr = args.lr * args.gamma))
    elif 'backbone' in components:
        optimizers.append(opt(model.backbone.parameters(), lr = args.lr * args.gamma))
    
    return optimizers
if __name__ == '__main__':
    main()
