import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets import MNISTDataset, CIFARDataset
from utils import get_args, accuracy
from meta_models import Model, to_var

from pdb import set_trace as bp
from tensorboardX import SummaryWriter
from torchvision import transforms
import copy

def main():

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

    if args.dataset == 'mnist':
        train_dataset = MNISTDataset(split = 'train', seed = args.seed)
        val_dataset = MNISTDataset(split = 'val', seed = args.seed)
        input_channel = 1
    elif args.dataset == 'cifar':
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
        train_dataset = CIFARDataset(split = 'train', seed = args.seed, transform = data_transforms['train'], percent = args.percent)
        val_dataset = CIFARDataset(split = 'val', seed = args.seed, transform = data_transforms['val'], percent = args.percent)
        input_channel = 3
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers)

    model = Model(input_channel = input_channel)
    optimizers = []
    for c in args.components:
        if c == 'all':
            optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
        elif c == 'fc':
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr = args.lr)
        elif c == 'backbone':
            optimizer = torch.optim.Adam(model.feature.parameters(), lr = args.lr)
        optimizers.append(optimizer)
        
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80], gamma=0.5, last_epoch=-1)

    if not os.path.exists(args.modeldir):
        os.mkdir(args.modeldir)
    writer = SummaryWriter(args.modeldir)

    best_prec = 0
    #model, optimizer, rollouts, current_optimizee_step, prev_optimizee_step = prepare_optimizee(args, input_channel, use_CUDA, args.num_steps, sgd_in_names, obs_shape, hidden_size, actor_critic, current_optimizee_step, prev_optimizee_step):
    for epoch in range(args.epochs):
        train(model, input_channel, optimizers, criterion, args.components, train_loader, val_loader, epoch, writer, use_CUDA, clamp = args.clamp)
        loss, prec = val(model, val_loader, criterion, epoch, writer, use_CUDA)
        torch.save(model, os.path.join(args.modeldir, 'checkpoint.pth.tar'))
        if prec > best_prec:
            torch.save(model, os.path.join(args.modeldir, 'model_best.pth.tar'))
            best_prec = prec


def train(model, input_channel, optimizers, criterion, components, train_loader, val_loader, epoch, writer, use_CUDA = True, clamp = False):
    model.train()
    accs = []
    losses = []
    losses_2 = []
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

        meta_model = Model(input_channel = input_channel)
        meta_model.load_state_dict(model.state_dict())
        if use_CUDA:
            meta_model = meta_model.cuda()

        input = to_var(input, requires_grad = False)
        label = to_var(label, requires_grad = False).long()
        try:
            val_input, val_label, _ = next(iter_val_loader)
        except:
            iter_val_loader = iter(val_loader)
            val_input, val_label, _ = next(iter_val_loader)

        val_input = to_var(val_input, requires_grad = False)
        val_label = to_var(val_label, requires_grad = False).long()
        
        y_f_hat = meta_model(input)
        cost = meta_criterion(y_f_hat, label)
        eps = to_var(torch.zeros(cost.size()))
        l_f_meta = (cost * eps).sum()
        meta_model.zero_grad()
        if 'all' in components:
            grads = torch.autograd.grad(l_f_meta, (meta_model.parameters()), create_graph=True)
            meta_model.update_params(0.001, source_params = grads)     

            y_g_hat = meta_model(val_input)
            l_g_meta = meta_criterion(y_g_hat, val_label).sum()
            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs = True)[0]
            if clamp:
                w1 = torch.clamp(-grad_eps, min = 0)
            else:
                w1 = -grad_eps
            
            norm_c = torch.sum(abs(w1))
            w1 = w1 / norm_c
            if ('fc' in components) or ('backbone' in components):
                w2 = copy.deepcopy(w1)
                w2[w2 > 0] = 0
                w1[w1 < 0] = 0

            w1_all.append(w1.detach().view(-1))
            if w2 is not None:
                w2_all.append(w2.detach().view(-1))
        
        else:
            grads_feature = torch.autograd.grad(l_f_meta, (meta_model.feature.parameters()), create_graph=True, retain_graph = True)
            grads_fc = torch.autograd.grad(l_f_meta, (meta_model.classifier.parameters()), create_graph=True)
            meta_model.feature.update_params(0.001, source_params = grads_feature)
            
            y_g_hat = meta_model(val_input)
            l_g_meta = meta_criterion(y_g_hat, val_label).sum()
            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs = True, retain_graph = True)[0]
            if clamp:
                w_1 = torch.clamp(-grad_eps, min = 0)
            else:
                w_1 = -grad_eps
            norm_c = torch.sum(abs(w_1))
            w_1 = w_1 / norm_c

            # FC backward
            meta_model.load_state_dict(model.state_dict())
            meta_model.classifier.update_params(0.001, source_params = grads_fc)

            y_g_hat = meta_model(val_input)
            l_g_meta = meta_criterion(y_g_hat, val_label).sum()
            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs = True, retain_graph = True)[0]
            
            if clamp:
                w_2 = torch.clamp(-grad_eps, min = 0)
            else:
                w_2 = -grad_eps
            norm_c = torch.sum(abs(w_2))

            w_2 = w_2 / norm_c
        
        index += 1
        output = model(input)
        loss = (meta_criterion(output, label) * w1).sum()
        #print(loss)
        prediction = torch.softmax(output, 1)

        optimizers[0].zero_grad()
        loss.backward(retain_graph = True)
        optimizers[0].step()
        
        if w2 is not None:
            loss_2 = (meta_criterion(output, label) * w2).sum()
            optimizers[1].zero_grad()
            loss_2.backward()
            optimizers[1].step()
            losses_2.append(loss_2.detach())

        top1 = accuracy(prediction, label)
        accs.append(top1)
        losses.append(loss.detach())
        

    acc = sum(accs) / len(accs)
    loss = sum(losses) / len(losses)
    if len(losses_2) > 0: loss_2 = sum(losses_2) / len(losses_2)
    else: loss_2 = 0
    w1_all = torch.cat(w1_all)
    w2_all = torch.cat(w2_all)
    noisy_labels = torch.cat(noisy_labels)
    true_labels = torch.cat(true_labels)
    writer.add_histogram("train/w1", w1_all, epoch)
    writer.add_histogram("train/w2", w2_all, epoch)
    print(noisy_labels)
    print(true_labels)
    writer.add_histogram("train/w1_on_noisy", w1_all[noisy_labels != true_labels], epoch)
    writer.add_histogram("train/w1_on_clean", w1_all[noisy_labels == true_labels], epoch)
    writer.add_histogram("train/w2_on_noisy", w2_all[noisy_labels != true_labels], epoch)
    writer.add_histogram("train/w2_on_clean", w2_all[noisy_labels == true_labels], epoch)

    writer.add_scalar("train/acc", acc, epoch)
    writer.add_scalar("train/loss", loss, epoch)
    if w2 is not None:
        writer.add_scalar("train/loss_2", loss_2, epoch)
    
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

if __name__ == '__main__':
    main()