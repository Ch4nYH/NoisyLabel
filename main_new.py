import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from datasets import MNISTDataset
from utils import get_args, accuracy
from meta_models import Model, to_var

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.models.policy import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from pdb import set_trace as bp
from tensorboardx import SummaryWriter

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
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = args.num_workers)

    iter_train_loader = iter(train_loader)
    iter_val_loader = iter(val_loader)

    model = Model(input_channel = input_channel)

    optimizer_backbone = torch.optim.Adam(model.feature.parameters(), lr = args.lr)
    optimizer_fc = torch.optim.Adam(model.fc.parameters(), lr = args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        [80], gamma=0.5, last_epoch=-1)

    if not os.path.exists(args.modeldir):
        os.mkdir(args.modeldir)
    writer = SummaryWriter(args.modeldir)

    best_prec = 0
    #model, optimizer, rollouts, current_optimizee_step, prev_optimizee_step = prepare_optimizee(args, input_channel, use_CUDA, args.num_steps, sgd_in_names, obs_shape, hidden_size, actor_critic, current_optimizee_step, prev_optimizee_step):
    for epoch in range(args.epochs):
        train(model, input_channel, optimizer_backbone, optimizer_fc, criterion, train_loader, val_loader, epoch, writer, use_CUDA)
        loss, prec = val(model, val_loader, criterion, epoch, writer, use_CUDA)
        torch.save(model, os.path.join(args.modeldir, 'checkpoint.pth.tar'))
        if prec > best_prec:
            torch.save(model, os.path.join(args.modeldir, 'model_best.pth.tar'))
            best_prec = prec


def train(model, input_channel, optimizer_backbone, optimizer_fc, criterion, train_loader, val_loader, epoch, writer, use_CUDA = True):
    model.train()
    accs = []
    losses = []
    iter_val_loader = iter(val_loader)
    meta_criterion = nn.CrossEntropyLoss(reduce = False)
    index = 0
    for (input, label) in train_loader:
        meta_model = Model(input_channel = input_channel)
        meta_model.load_state_dict(model.state_dict())
        if use_CUDA:
            meta_model = meta_model.cuda()

        input = to_var(input, requires_grad = False)
        label = to_var(label, requires_grad = False).long()
        y_f_hat = meta_model(input)
        cost = meta_criterion(y_f_hat, label)
        eps = to_var(torch.zeros(cost.size()))
        l_f_meta = (cost * eps).sum()
        meta_model.zero_grad()

        # Backbone backward
        grads = torch.autograd.grad(l_f_meta, (meta_model.feature.parameters()), create_graph=True)
        meta_model.update_params(0.001, source_params = grads)
        try:
            val_input, val_label = next(iter_val_loader)
        except:
            iter_val_loader = iter(val_loader)
            val_input, val_label = next(iter_val_loader)

        val_input = to_var(val_input, requires_grad = False)
        val_label = to_var(val_label, requires_grad = False).long()

        y_g_hat = meta_model(val_input)
        l_g_meta = meta_criterion(y_g_hat, val_label).sum()
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs = True)[0]
        if index % 100 == 0:
            print("[{}/{}] BB Positive: {}, Negative: {}" .format(index, len(train_loader), torch.sum(grad_eps > 0), torch.sum(grad_eps < 0)))
        #w = torch.clamp(-grad_eps, min = 0)
        w_backbone = -grad_eps
        norm_c = torch.sum(abs(w_backbone))
        w_backbone = w_backbone / norm_c

        # FC backward
        grads = torch.autograd.grad(l_f_meta, (meta_model.fc.parameters()), create_graph=True)
        meta_model.update_params(0.001, source_params = grads)
        try:
            val_input, val_label = next(iter_val_loader)
        except:
            iter_val_loader = iter(val_loader)
            val_input, val_label = next(iter_val_loader)

        val_input = to_var(val_input, requires_grad = False)
        val_label = to_var(val_label, requires_grad = False).long()

        y_g_hat = meta_model(val_input)
        l_g_meta = meta_criterion(y_g_hat, val_label).sum()
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs = True)[0]
        if index % 100 == 0:
            print("[{}/{}] FC Positive: {}, Negative: {}" .format(index, len(train_loader), torch.sum(grad_eps > 0), torch.sum(grad_eps < 0)))
        index += 1
        #w = torch.clamp(-grad_eps, min = 0)
        w_fc = -grad_eps
        norm_c = torch.sum(abs(w_fc))

        w_fc = w_fc / norm_c

        output = model(input)
        prediction = torch.softmax(output, 1)

        backbone_loss = (meta_criterion(output, label) * w_backbone).sum()
        #print(loss)
        optimizer_backbone.zero_grad()
        if backbone_loss < 10000:
            backbone_loss.backward(retain_graph = True)
            optimizer_backbone.step()
        else:
            bp()

        fc_loss = (meta_criterion(output, label) * w_fc).sum()
        #print(loss)
        optimizer_fc.zero_grad()
        if fc_loss < 10000:
            fc_loss.backward()
            optimizer_fc.step()
        else:
            bp()


        top1 = accuracy(prediction, label)
        accs.append(top1)
        losses.append(loss.detach())

    acc = sum(accs) / len(accs)
    loss = sum(losses) / len(losses)
    writer.add_scalar("train/acc", acc, epoch)
    writer.add_scalar("train/loss", loss, epoch)
    print("Training Epoch: {}, Accuracy: {}, Losses: {}".format(epoch, acc, loss))
    return acc, loss

def val(model, val_loader, criterion, epoch, writer, use_CUDA = True):
    model.eval()
    accs = []
    losses = []
    with torch.no_grad():
        for (input, label) in val_loader:
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
    writer.add_scalar("train/acc", acc, epoch)
    writer.add_scalar("train/loss", loss, epoch)
    print("Validation Epoch: {}, Accuracy: {}, Losses: {}".format(epoch, acc, loss))
    return acc, loss

if __name__ == '__main__':
    main()